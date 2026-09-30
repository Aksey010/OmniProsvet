import threading
import time
from pathlib import Path

from loguru import logger

from src.configuration import settings

CHAT_URL = "https://chat.deepseek.com/"
PROFILE_DIR = Path(__file__).resolve().parent.parent / "data" / "browser-profile"

ANSWER_TIMEOUT = 600
POLL_INTERVAL = 2
MANUAL_LOGIN_WAIT = 300

TIMEOUT_TEXT = "Таймаут ожидания ответа DeepSeek"


class GenerationError(RuntimeError):
    pass


class DeepSeekAgent:
    def __init__(self, profile_dir: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._driver = None
        self._profile_dir = profile_dir or PROFILE_DIR
        self._last_call = 0.0

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    def _ensure_driver(self):
        if self._driver is not None:
            return self._driver
        try:
            return self._start_driver()
        except GenerationError:
            logger.warning(
                "Запуск браузера не удался — чищу зависшие процессы Edge/Chrome с нашим профилем и пробую ещё раз…"
            )
            self._kill_stale_browsers()
            time.sleep(2)
            return self._start_driver()

    def _start_driver(self):
        from selenium import webdriver

        self._profile_dir.mkdir(parents=True, exist_ok=True)
        errors = []
        for browser in ("edge", "chrome"):
            try:
                self._driver = webdriver.Edge(options=self._build_options("edge")) \
                    if browser == "edge" \
                    else webdriver.Chrome(options=self._build_options("chrome"))
                logger.info(f"Браузер для генерации: {browser}")
                return self._driver
            except Exception as exc:
                errors.append(f"{browser}: {exc}")
        raise GenerationError("Не удалось запустить браузер (нужен Edge или Chrome): " + "; ".join(errors))

    @staticmethod
    def _kill_stale_browsers() -> None:
        """Убивает зависшие драйверы и браузеры с нашим профилем (не трогает обычные окна Edge/Chrome)."""
        import subprocess

        ps_cmd = (
            "Get-CimInstance Win32_Process "
            "-Filter \"Name='msedgedriver.exe' OR Name='chromedriver.exe'\" "
            "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; "
            "Get-CimInstance Win32_Process "
            "-Filter \"Name='msedge.exe' OR Name='chrome.exe'\" "
            "| Where-Object { $_.CommandLine -like '*browser-profile*' } "
            "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True,
                timeout=30,
            )
        except Exception as exc:
            logger.warning(f"Не удалось почистить зависшие браузеры: {exc}")

    def _build_options(self, browser: str):
        from selenium import webdriver

        options_cls = webdriver.EdgeOptions if browser == "edge" else webdriver.ChromeOptions
        opts = options_cls()
        opts.add_argument(f"--user-data-dir={self._profile_dir}")
        opts.add_argument("--window-size=1100,900")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        if settings.deepseek_headless:
            opts.add_argument("--headless=new")
        if settings.deepseek_hidden_window:
            # окно за пределами экрана: его не видно, но Chrome считает его видимым
            opts.add_argument("--window-position=-32000,-32000")
        # запрет «засыпания» страницы при сворачивании/перекрытии окна
        opts.add_argument("--disable-backgrounding-occluded-windows")
        opts.add_argument("--disable-renderer-backgrounding")
        opts.add_argument("--disable-background-timer-throttling")
        opts.add_argument("--disable-features=CalculateNativeWinOcclusion")
        opts.add_argument("--disable-hang-monitor")
        opts.add_argument("--disable-ipc-flooding-protection")
        opts.add_argument("--mute-audio")
        return opts

    def _logged_in(self, driver) -> bool:
        url = driver.current_url.lower()
        if "sign_in" in url or "login" in url:
            return False
        return not driver.find_elements("css selector", "input[type=password]")

    def _close_extra_tabs(self, driver) -> None:
        try:
            handles = driver.window_handles
            for handle in handles[1:]:
                driver.switch_to.window(handle)
                driver.close()
            driver.switch_to.window(handles[0])
        except Exception:
            pass

    def _wait_visible(self, driver, css: str, timeout: int = 20) -> list:
        deadline = time.time() + timeout
        elements = []
        while time.time() < deadline:
            try:
                elements = [el for el in driver.find_elements("css selector", css) if el.is_displayed()]
            except Exception:
                elements = []
            if elements:
                return elements
            time.sleep(1)
        return elements

    def _try_autologin(self, driver, login: str, password: str) -> bool:
        if not login or not password:
            return False
        try:
            inputs = self._wait_visible(driver, "input[type=text], input[type=email]", 15)
            pwd_inputs = self._wait_visible(driver, "input[type=password]", 5)
            if not inputs or not pwd_inputs:
                return False
            inputs[0].clear()
            inputs[0].send_keys(login)
            time.sleep(0.5)
            pwd_inputs[0].clear()
            pwd_inputs[0].send_keys(password)
            time.sleep(0.5)

            clicked = False
            deadline = time.time() + 15
            while time.time() < deadline and not clicked:
                for btn in driver.find_elements("css selector", 'button, div[role="button"]'):
                    try:
                        if not btn.is_displayed():
                            continue
                        label = " ".join((btn.text or "").split())
                        if label in ("Войти", "Log In", "Login", "登录"):
                            btn.click()
                            clicked = True
                            logger.info("Нажата кнопка входа DeepSeek")
                            break
                    except Exception:
                        continue
                if not clicked:
                    time.sleep(1)
            if not clicked:
                logger.warning("Кнопка входа DeepSeek не найдена")
                try:
                    shot = self._profile_dir.parent / "debug_login_page.png"
                    driver.save_screenshot(str(shot))
                    body_text = driver.execute_script("return document.body ? document.body.innerText : ''") or ""
                    logger.info(f"Диагностика[логин]: скриншот={shot} | страница: {body_text[:300].replace(chr(10), ' | ')}")
                except Exception:
                    pass
                return False

            for _ in range(15):
                time.sleep(2)
                try:
                    if self._logged_in(driver):
                        logger.info("Автологин в DeepSeek выполнен")
                        return True
                except Exception:
                    pass
            try:
                driver.save_screenshot(str(self._profile_dir.parent / "debug_login.png"))
                logger.warning(
                    f"После клика вход не произошёл, скриншот: {self._profile_dir.parent / 'debug_login.png'}"
                )
            except Exception:
                pass
            return False
        except Exception as exc:
            logger.warning(f"Автологин не удался: {exc}")
            return False

    def _ensure_logged_in(self, driver, login: str, password: str) -> None:
        self._close_extra_tabs(driver)
        driver.get(CHAT_URL)
        time.sleep(3)
        if self._logged_in(driver):
            return
        if self._try_autologin(driver, login, password):
            driver.get(CHAT_URL)
            time.sleep(2)
            if self._logged_in(driver):
                return
        logger.info(
            f"Жду ручной вход в DeepSeek в открывшемся окне браузера "
            f"(до {MANUAL_LOGIN_WAIT} сек)…"
        )
        deadline = time.time() + MANUAL_LOGIN_WAIT
        while time.time() < deadline:
            time.sleep(3)
            try:
                if self._logged_in(driver):
                    logger.info("Вход выполнен, продолжаю")
                    driver.get(CHAT_URL)
                    time.sleep(2)
                    return
            except Exception:
                pass
        raise GenerationError("Не дождался входа в DeepSeek — войдите вручную и повторите")

    _SET_VALUE_JS = """
const el = arguments[0], val = arguments[1];
if (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT') {
    const proto = el.tagName === 'TEXTAREA'
        ? window.HTMLTextAreaElement.prototype
        : window.HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, val);
} else {
    el.innerText = val;
}
el.dispatchEvent(new Event('input', {bubbles: true}));
el.dispatchEvent(new Event('change', {bubbles: true}));
"""

    _EXTRACT_JS = """
const nodes = document.querySelectorAll('.ds-markdown');
if (!nodes.length) return '';
return nodes[nodes.length - 1].innerText || '';
"""

    _SHOW_FILE_INPUT_JS = """
const el = arguments[0];
el.style.display = 'block';
el.style.visibility = 'visible';
el.style.position = 'fixed';
el.style.left = '0';
el.style.top = '0';
el.style.opacity = '0.01';
"""

    _DOM_SIZE_JS = """
return document.body ? document.body.innerHTML.length : 0;
"""

    _FIND_SEND_BTN_JS = """
const input = document.querySelector('textarea#chat-input') || document.querySelector('textarea') || document.querySelector('div[contenteditable="true"]');
if (!input) return null;
let root = input.closest('form') || input.parentElement;
let last = null;
for (let i = 0; i < 4 && root; i++) {
  const btns = root.querySelectorAll('button, div[role="button"]');
  for (const el of btns) {
    const dis = el.disabled || el.getAttribute('aria-disabled') === 'true';
    if (dis) {
      const r = el.getBoundingClientRect();
      if (r.width > 0 && r.height > 0) last = el;
    }
  }
  root = root.parentElement;
}
return last;
"""

    _BTN_STATE_JS = """
const el = arguments[0];
if (!el) return 'gone';
if (el.disabled) return 'disabled';
if (el.getAttribute('aria-disabled') === 'true') return 'disabled';
const r = el.getBoundingClientRect();
if (r.width === 0 || r.height === 0) return 'hidden';
return 'ready';
"""

    def _attach_images(self, driver, paths: list[str], timeout: int | None = None) -> None:
        file_input = None
        for css in ("input[type=file]",):
            elements = driver.find_elements("css selector", css)
            for el in elements:
                file_input = el
                break
            if file_input:
                break
        if not file_input:
            raise GenerationError("Не нашёл кнопку вложения файлов в чате DeepSeek")

        total_mb = 0.0
        for p in paths:
            try:
                total_mb += Path(p).stat().st_size / 1048576
            except OSError:
                pass
        if timeout is None:
            timeout = max(120, int(total_mb * 6))

        try:
            before = driver.execute_script(self._DOM_SIZE_JS) or 0
        except Exception:
            before = 0

        driver.execute_script(self._SHOW_FILE_INPUT_JS, file_input)
        joined = "\n".join(paths)
        file_input.send_keys(joined)
        logger.info(
            f"Прикладываю {len(paths)} файл(ов) ({total_mb:.1f} МБ), "
            f"жду обработку до активации кнопки отправки (до {timeout}с)…"
        )

        send_btn = None
        for _ in range(10):
            try:
                send_btn = driver.execute_script(self._FIND_SEND_BTN_JS)
            except Exception:
                send_btn = None
            if send_btn is not None:
                break
            time.sleep(1)

        deadline = time.time() + timeout
        warned = False
        last_status = ""
        while time.time() < deadline:
            time.sleep(1)
            status = ""
            if send_btn is not None:
                try:
                    status = driver.execute_script(self._BTN_STATE_JS, send_btn)
                except Exception:
                    try:
                        send_btn = driver.execute_script(self._FIND_SEND_BTN_JS)
                    except Exception:
                        send_btn = None
                    status = "переопределена" if send_btn is not None else ""
            if status == "ready":
                time.sleep(2)
                logger.info("Вложения обработаны, кнопка отправки активна")
                return
            if status and status != last_status:
                last_status = status
                logger.info(f"Ожидаю вложения: {status}")
            if not warned and time.time() - (deadline - timeout) >= 60 and not status:
                warned = True
                logger.warning("Кнопка отправки не отслеживается — полагаюсь на таймаут")
        logger.warning("Вложения не дождались (таймаут) — отправляю как есть")

    def _set_mode(self, driver, label: str) -> bool:
        try:
            for el in driver.find_elements("css selector", 'div[role="button"], button, div, span'):
                try:
                    if not el.is_displayed():
                        continue
                    text = " ".join((el.text or "").split())
                    if text == label and el.tag_name.lower() != "textarea":
                        el.click()
                        logger.info(f"Выбран режим DeepSeek: {label}")
                        time.sleep(1)
                        return True
                except Exception:
                    continue
        except Exception as exc:
            logger.warning(f"Не удалось выбрать режим {label}: {exc}")
        return False

    def _set_toggle(self, driver, label: str, want_on: bool) -> None:
        try:
            for el in driver.find_elements("css selector", "div.ds-toggle-button"):
                try:
                    if not el.is_displayed():
                        continue
                    text = " ".join((el.text or "").split())
                    if text != label:
                        continue
                    is_on = "ds-toggle-button--selected" in (el.get_attribute("class") or "")
                    if is_on != want_on:
                        el.click()
                        time.sleep(1)
                        logger.info(f"Тогл DeepSeek «{label}» → {'вкл' if want_on else 'выкл'}")
                    else:
                        logger.info(f"Тогл DeepSeek «{label}» уже {'включён' if want_on else 'выключен'}")
                    return
                except Exception:
                    continue
            logger.warning(f"Тогл «{label}» не найден")
        except Exception as exc:
            logger.warning(f"Ошибка работы с тоглом {label}: {exc}")

    def _find_input(self, driver):
        for css in ("textarea#chat-input", "textarea", 'div[contenteditable="true"]'):
            elements = driver.find_elements("css selector", css)
            for el in elements:
                if el.is_displayed():
                    return el
        return None

    def _click_send(self, driver, input_el) -> bool:
        from selenium.webdriver.common.keys import Keys

        for css in (
            'button[type="submit"]',
            'div[role="button"][aria-disabled="false"]',
        ):
            for btn in driver.find_elements("css selector", css):
                try:
                    if btn.is_displayed() and btn.is_enabled():
                        btn.click()
                        return True
                except Exception:
                    continue
        try:
            input_el.send_keys(Keys.ENTER)
            return True
        except Exception:
            return False

    def _quit_driver(self) -> None:
        if self._driver is not None:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None

    def reset_profile(self) -> None:
        with self._lock:
            self._quit_driver()
            import shutil

            try:
                if self._profile_dir.exists():
                    shutil.rmtree(self._profile_dir, ignore_errors=True)
                logger.info(f"Профиль браузера сброшен: {self._profile_dir.name}")
            except Exception as exc:
                logger.warning(f"Не удалось сбросить профиль: {exc}")

    def _pace(self) -> None:
        import random

        wait = random.uniform(
            max(0, settings.deepseek_pause_min), max(0, settings.deepseek_pause_max)
        )
        since = time.time() - self._last_call
        if self._last_call and since < wait:
            remaining = wait - since
            if remaining > 1:
                logger.info(f"Пауза между запросами DeepSeek: {remaining:.0f} сек")
            time.sleep(remaining)
        self._last_call = time.time()

    def send_prompt(
        self,
        prompt: str,
        login: str = "",
        password: str = "",
        deepthink: bool = True,
        expert: bool = True,
        attachments: list[str] | None = None,
        stable_rounds: int = 12,
        model: str | None = None,
    ) -> str:
        from selenium.common.exceptions import WebDriverException

        with self._lock:
            self._pace()
            for attempt in (1, 2, 3):
                try:
                    return self._send_prompt_once(
                        prompt, login, password, deepthink, expert, attachments, stable_rounds
                    )
                except GenerationError as exc:
                    if TIMEOUT_TEXT in str(exc) and attempt < 3:
                        logger.warning(f"{exc} — повторная попытка {attempt + 1}/3")
                        self._quit_driver()
                        continue
                    raise
                except WebDriverException as exc:
                    self._quit_driver()
                    if attempt == 3:
                        raise GenerationError(f"Браузер не отвечает: {exc}")
                    logger.warning(f"Сессия браузера умерла ({type(exc).__name__}), перезапускаю…")

    _REJECT_JS = """
return document.body ? document.body.innerText : '';
"""

    def _check_rejected(self, driver) -> str:
        try:
            text = driver.execute_script(self._REJECT_JS) or ""
        except Exception:
            return ""
        lowered = text.lower()
        if "message not sent" in lowered or "content violates terms" in lowered:
            return "DeepSeek отклонил сообщение (нарушение правил платформы) — вложение помечено «May violate terms»"
        return ""

    def _send_prompt_once(
        self,
        prompt: str,
        login: str,
        password: str,
        deepthink: bool,
        expert: bool,
        attachments: list[str] | None = None,
        stable_rounds: int = 12,
    ) -> str:
        driver = self._ensure_driver()
        self._ensure_logged_in(driver, login, password)

        input_el = None
        for _ in range(15):
            input_el = self._find_input(driver)
            if input_el:
                break
            time.sleep(1)
        if not input_el:
            raise GenerationError("Не нашёл поле ввода чата DeepSeek")

        if attachments:
            self._attach_images(driver, attachments)

        if expert:
            self._set_mode(driver, "Эксперт")
        # тогл выставляем явно в оба состояния — сохранённое состояние профиля не должно влиять
        self._set_toggle(driver, "Глубокое мышление", bool(deepthink))

        driver.execute_script(self._SET_VALUE_JS, input_el, prompt)
        time.sleep(0.5)
        if not self._click_send(driver, input_el):
            raise GenerationError("Не смог нажать кнопку отправки")
        logger.info(f"Промпт отправлен в DeepSeek ({len(prompt)} символов), жду ответ…")

        deadline = time.time() + ANSWER_TIMEOUT
        last_text = ""
        stable = 0
        last_progress = time.time()
        while time.time() < deadline:
            time.sleep(POLL_INTERVAL)
            rejected = self._check_rejected(driver)
            if rejected:
                self._dump_page(driver, "rejected")
                raise GenerationError(rejected)
            try:
                current = driver.execute_script(self._EXTRACT_JS) or ""
            except Exception:
                continue
            if current == last_text:
                stable += 1
            else:
                stable = 0
                last_text = current
            # DeepSeek иногда обрывает выдачу — после паузы проверяем кнопку «Продолжить»
            if stable == 5:
                try:
                    clicked = driver.execute_script(self._CLICK_CONTINUE_JS)
                    if clicked:
                        logger.info(f"Проверяю кнопку «{clicked}» в DeepSeek (текст стабилен)")
                except Exception:
                    pass
            elapsed = int(ANSWER_TIMEOUT - (deadline - time.time()))
            if time.time() - last_progress >= 30:
                last_progress = time.time()
                vis = ""
                try:
                    vis = f" visibility={driver.execute_script('return document.visibilityState + chr(47) + String(document.hidden)')}"
                except Exception:
                    pass
                logger.info(f"DeepSeek: жду ответ {elapsed}/{ANSWER_TIMEOUT}с, в тексте {len(last_text)} символов{vis}")
                if not last_text and elapsed >= 180:
                    self._dump_page(driver, "stuck")
                    raise GenerationError(
                        f"{TIMEOUT_TEXT} — после {elapsed}с нет ни одного символа ответа (похоже, запрос не ушёл или вкладка не активна)"
                    )
            # текст стабилен stable_rounds*2 секунд — считаем ответ завершённым,
            # даже если на странице осталась кнопка «Продолжить»
            if stable >= max(1, stable_rounds) and len(last_text) > 0:
                logger.info(
                    f"Ответ DeepSeek получен ({len(last_text)} символов)"
                    f" за {elapsed} сек"
                )
                return last_text.strip()
        self._dump_page(driver, "timeout")
        raise GenerationError(f"{TIMEOUT_TEXT} (прошло {ANSWER_TIMEOUT}с, последний текст {len(last_text)} симв.)")

    _CLICK_CONTINUE_JS = """
const labels = ['Продолжить', 'Continue', '继续'];
const sel = 'button, div[role="button"], a';
const nodes = document.querySelectorAll(sel);
for (const el of nodes) {
  const t = (el.innerText || '').trim();
  if (labels.includes(t)) { el.click(); return t; }
}
return '';
"""

    def _dump_page(self, driver, tag: str) -> None:
        """Скриншот + текст страницы + состояние видимости — диагностика зависаний."""
        try:
            vis = "unknown"
            try:
                vis = str(driver.execute_script(
                    "return (document.visibilityState||'?') + '/' + String(document.hidden)"
                ))
            except Exception:
                pass
            logger.info(f"Диагностика[{tag}]: url={driver.current_url[:80]} | visibility={vis}")
            stamp = time.strftime("%Y%m%d_%H%M%S")
            screenshot = self._profile_dir.parent / f"debug_{tag}_{stamp}.png"
            driver.save_screenshot(str(screenshot))
            body_text = driver.execute_script("return document.body ? document.body.innerText : ''") or ""
            sample = body_text[:1200].replace("\n", " | ")
            logger.info(f"Диагностика[{tag}]: скриншот={screenshot} | текст страницы: {sample}")
        except Exception as exc:
            logger.warning(f"Диагностика[{tag}] не удалась: {exc}")


agent = DeepSeekAgent()
agent2 = DeepSeekAgent(PROFILE_DIR.parent / "browser-profile-2")
