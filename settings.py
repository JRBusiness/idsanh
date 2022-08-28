import errno
import os
from os.path import basename

import cv2
from dotenv import load_dotenv

load_dotenv()


def get_env(variable_name, default=None):
    value = os.getenv(variable_name)
    if value is None:
        if default is None:
            raise ValueError(f"{variable_name} is not presented in environment variables. Check your .env file")
        else:
            return default
    if str(value).lower() in {"true", "false"}:
        return str(value).lower() == "true"
    return value


# setup log base path
def setup_log_base_path(path):
    """
    This method will create a log directory if it doesnt exist
    In the case that it's not running in docker it places it adjacent to the server directory in this project.
    In the case that it is a docker container it places it in "/var/log/id-scanner-flask" which is managed by logrotate
    :return: full directory path for logs
    """
    try:
        os.makedirs(path)
        return path
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            return path
        if basename(os.getcwd()) == "server":
            os.chdir("")
        return setup_log_base_path(os.path.join(os.getcwd(), "logs"))


class Config:
    app_name: str = os.getenv("APP_NAME", "id_scanner_api")
    version: float = os.getenv("VERSION", 0.4)
    port: int = int(os.getenv("FLASK_RUN_PORT"))
    host: str = os.getenv("FLASK_RUN_HOST")
    debug: bool = os.getenv("FLASK_DEBUG", False)
    fastapi_key: str = os.getenv("FASTAPI_KEY")
    fastapi_salt: str = os.getenv("SALT")
    fastapi_secret: str = os.getenv("SECRET")
    auth_expire: int = os.getenv("AUTH_EXPIRE")
    LOG_BASE = setup_log_base_path("/var/log/id-scanner")
    barcode_app_name: str = "barcode_scan_app"
    insurance_app_name: str = "insurance__scan_app"
    id_app_name: str = "id_scan_app"
    single_m: float = 0.03
    multi_m: float = 0.07

    alphabet: str = ' "%&\'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyzÀÁÂÃÉÊÌÍÒÓÔÙÚÝàáâãèéêìíòóôõùúýĂăĐđĩŨũƠơƯưẠạẢảẤấẦầẨẩẫẬậẮắẰằẳẶặẹẺẻẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỶỷỸỹ'
    node_labels: list = ['OTHER', 'ADDRESS', 'SELLER', 'TIMESTAMP', 'TOTAL_COST']
    kie_config = "server/shared/helpers/kie/"
    kie_weight = "server/shared/helpers/kie"

    u2net_th = 0.5
    score_th = 0.82
    get_max = True
    merge_text = True
    visualize = False


config = Config()
