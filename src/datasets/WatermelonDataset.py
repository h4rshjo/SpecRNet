
import logging
from pathlib import Path

import pandas as pd

from src.datasets.base_dataset import SimpleAudioFakeDataset

DF_ASVSPOOF_SPLIT = {
    "partition_ratio": [0.7, 0.15],
    "seed": 45
}

LOGGER = logging.getLogger()

# "label", "bonafied", "spoof" sa column
class WatermelonDataset(SimpleAudioFakeDataset):

    def __init__(self, path, subset="train", transform=None):
        super().__init__(subset, transform)
        self.path = path

        self.partition_ratio = DF_ASVSPOOF_SPLIT["partition_ratio"]
        self.seed = DF_ASVSPOOF_SPLIT["seed"]

        self.samples = self.read_protocol()

        self.transform = transform
        LOGGER.info(f"Spoof: {len(self.samples[self.samples['label'] == 'spoof'])}")
        LOGGER.info(f"Original: {len(self.samples[self.samples['label'] == 'bonafide'])}")

    def read_protocol(self):
        samples = {
            "sample_name": [],
            "label": [],
            "path": [],
            "attack_type": [],
        }

        real_samples = []
        fake_samples = []
        # Given that the path is: "./Bonafide_Dataset/train/*" grab all the .wav files
        # and store them in the samples dictionary
        # Check if the file exists and is readable
        import os
        if not os.path.exists(self.path):
            logging.error(f"File does not exist: {self.path}")
            return
        if not os.access(self.path, os.R_OK):
            logging.error(f"File is not readable: {self.path}")
            return
        
        ## Iterate through the directory of self.path
        for file in os.listdir(self.path):
            if file.endswith(".wav"):
                # Get the label of the sample
                label = file.split("$$")[1]
                label = label.split(".")[0]

                # Add the sample to the correct list
                if label == "bonafide":
                    real_samples.append(file)
                elif label == "spoof":
                    fake_samples.append(file)

        logging.info(f"Real samples: {len(real_samples)}")

        fake_samples = self.split_samples(fake_samples)
        logging.info("watermelon")
        for line in fake_samples:
            samples = self.add_line_to_samples(samples, line)

        real_samples = self.split_samples(real_samples)
        for line in real_samples:
            samples = self.add_line_to_samples(samples, line)

        df = pd.DataFrame(samples)
        print (df)
        LOGGER.info(f"Constructed DataFrame with {len(df)} samples.")
        return df

    def add_line_to_samples(self, samples, line):
        sample_name, label = line.split("$$")
        samples["sample_name"].append(sample_name)
        samples["label"].append(label.split(".")[0])
        samples["attack_type"].append(label)
        samples["path"].append(Path(self.path) / line)

        return samples

