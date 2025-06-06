from server_deployment import ServerDeployment
from kubernetes import config
from logger import get_logger, configure_structlog
from service_discovery import Service
import yaml
import argparse

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
    
    def get_triton_deployment(self):
        self.triton_deployment = ServerDeployment(self.release_name, self.namespace)
    
    def get_servers(self):
        self.get_triton_deployment()
        return self.triton_deployment.get_servers()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Triton server model loader')
    parser.add_argument('-d', '--debug', action='store_true',
                      help='Enable debug logging')
    args = parser.parse_args()

    # Set global debug mode
    configure_structlog(args.debug)    
    logger = get_logger("main")
    logger.info("Starting application", 
                release=release_name,
                namespace=namespace,
                models=models)
    
    app = App(release_name, namespace)
    app.init_services(models)

    # app.get_triton_deployment()
    # app.triton_deployment.scale(2)

    servers = app.get_servers()
    for i, server in enumerate(servers):
        server.logger.info(f"Processing server {i}", pod=server.pod_name)
        # server.sync_labels()
        # server.remove_label("deepmet-v1")
        if i==0:
            server.logger.warning("Will unload model from this server", pod=server.pod_name, model="deepmet")
            server.unload_model("deepmet")
        else:
            server.logger.warning("Will load model into this server", pod=server.pod_name, model="deepmet")
            server.load_model("deepmet")
        # server.restart()
        # server.get_gpu_memory()
        server.get_models()
    app.triton_deployment.get_aggregated_model_repository_index()

    logger.info("Done!")