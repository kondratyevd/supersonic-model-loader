from server_deployment import ServerDeployment
from kubernetes import config
from logger import get_logger
from service_discovery import Service
import yaml

# Load configuration from config.yaml
with open('config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)
    release_name = config_data.get('release_name', 'supersonic')
    namespace = config_data.get('namespace', 'cms')
    models = config_data.get('models', ['deepmet-v1'])

class App:
    def __init__(self, release_name: str, namespace: str):
        self.release_name = release_name
        self.namespace = namespace
        self.logger = get_logger("app")
        self.services = {}
        
        try:
            config.load_kube_config()
            self.logger.info("Loaded kube config from local environment")
        except config.ConfigException:
            config.load_incluster_config()
            self.logger.info("Loaded kube config from in-cluster environment")

    def init_services(self, models):
        for model_name_full in models:
            self.spawn_service(model_name_full)

    def spawn_service(self, model_name_full: str):
        service = Service(model_name_full, self.release_name, self.namespace)
        service.spawn()        
        self.services[model_name_full] = service
        return service
    
    def get_servers(self):
        """
        Use kubectl to find any existing Triton servers
        """
        server_deployment = ServerDeployment(self.release_name, self.namespace)
        deployment = server_deployment.get_deployment()
        return server_deployment.get_servers()

if __name__ == "__main__":
    logger = get_logger("main")
    logger.info("Starting application", 
                release=release_name,
                namespace=namespace,
                models=models)
    
    app = App(release_name, namespace)
    app.init_services(models)

    servers = app.get_servers()
    for server in servers:
        # server.sync_labels()
        # server.unload_model("deepmet")
        server.get_models()

        # print(server.get_gpu_memory())

    logger.info("Application startup completed")