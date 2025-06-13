import os
import sys
import pandas as pd
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath('./src'))
from preprocessing import run_preproc, load_train_data_with_enc
from scorer import make_pred, get_feature_import, get_probability

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(self):
        logger.info('Initializing ProcessingService...')
        self.input_dir = '/app/input'
        self.output_dir = '/app/output'
        self.train, self.enc = load_train_data_with_enc()
        logger.info('Service initialized')

    def process_single_file(self, file_path):
        try:
            
            logger.info('Processing file: %s', file_path)
            time.sleep(15)
            input_df = pd.read_csv(file_path)

            logger.info('Starting preprocessing')
            processed_df = run_preproc(self.train, input_df, self.enc)
            
            logger.info('Making prediction')
            submission = make_pred(processed_df, file_path)

            
            logger.info('Prepraring submission file')
            timestamp = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
            output_filename = f"sample_submission_{timestamp}_{os.path.basename(file_path)}"
            submission.to_csv(os.path.join(self.output_dir, output_filename), index=False)
            logger.info('Predictions saved to: %s', output_filename)

            logger.info('Creating JSON file with FI')
            top_5_features = get_feature_import(file_path)

            logger.info('Prepraring JSON file')
            timestamp = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
            output_filename = f"top5_features_{timestamp}.json"
            with open(os.path.join(self.output_dir, output_filename), 'w') as json_file:
                json.dump(top_5_features, json_file)
            logger.info('JSON saved to: %s', output_filename)

            logger.info('Creating Image file with FI')
            proba = get_probability(processed_df)

            logger.info('Prepraring PNG file')
            timestamp = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
            output_filename = f"proba_image_{timestamp}.png"
            plt.figure(figsize=(10, 6))
            sns.set(style="whitegrid")

            sns.histplot(proba, stat="density", bins=20, kde=True)
            plt.xlabel("Probability score")
            plt.ylabel("Density")
            plt.savefig(os.path.join(self.output_dir, output_filename), format='png', dpi=300, bbox_inches='tight')
            logger.info('Image file saved to: %s', output_filename)

            

        except Exception as e:
            logger.error('Error processing file %s: %s', file_path, e, exc_info=True)
            return


class FileHandler(FileSystemEventHandler):
    def __init__(self, service):
        self.service = service

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".csv"):
            logger.debug('New file detected: %s', event.src_path)
            self.service.process_single_file(event.src_path)

if __name__ == "__main__":
    logger.info('Starting ML scoring service...')
    service = ProcessingService()
    observer = Observer()
    observer.schedule(FileHandler(service), path=service.input_dir, recursive=False)
    observer.start()
    logger.info('File observer started')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
        observer.stop()
    observer.join()