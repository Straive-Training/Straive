import dotenv
import os
import json
import mysql.connector


class ENVIRONMENT:
    def __init__(self):
        project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
        dotenv_path = os.path.join(project_dir, '.env')
        dotenv.load_dotenv(dotenv_path)
        self.domain = os.getenv("DOMAIN")
        self.port = os.getenv("PORT")
        self.prefix = os.getenv("PREFIX")

        # DB settings
        self.db_host = os.getenv("DB_HOST")
        self.db_port = int(os.getenv("DB_PORT", 3306))
        self.db_user = os.getenv("DB_USER")
        self.db_password = os.getenv("DB_PASSWORD")
        self.db_name = os.getenv("DB_NAME")

    def get_db_config(self, include_db=True):
        config = {
            "host": self.db_host,
            "port": self.db_port,
            "user": self.db_user,
            "password": self.db_password,
        }
        if include_db:
            config["database"] = self.db_name
        return config

    def get_instance(self):
        if not hasattr(self, "_instance"):
            self._instance = ENVIRONMENT()
        return self._instance

    def getDomain(self):
        return self.domain

    def getPort(self):
        return self.port

    def getPrefix(self):
        return self.prefix


# Global instance to avoid multiple .env loads
env_instance = ENVIRONMENT().get_instance()

domain = env_instance.getDomain()
port = env_instance.getPort()
prefix = env_instance.getPrefix()


def get_db_connection():
    """Get MySQL connection with database."""
    config = env_instance.get_db_config(include_db=True)
    return mysql.connector.connect(**config)


def get_connection_no_db():
    """Get MySQL connection without specifying database (used for creating DB)."""
    config = env_instance.get_db_config(include_db=False)
    return mysql.connector.connect(**config)


def build_swagger_config_json():
    config_file_path = 'static/swagger/config.json'

    with open(config_file_path, 'r') as file:
        config_data = json.load(file)

    config_data['servers'] = [
        {"url": f"http://localhost:{port}{prefix}"},
        {"url": f"http://{domain}:{port}{prefix}"}
    ]

    with open(config_file_path, 'w') as new_file:
        json.dump(config_data, new_file, indent=2)
