import logging
from typing import List, Optional


import pandas as pd

from src.datasets.base_dataset import SimpleAudioFakeDataset
from src.datasets.deepfake_asvspoof_dataset import DeepFakeASVSpoofDataset
from src.datasets.fakeavceleb_dataset import FakeAVCelebDataset
from src.datasets.wavefake_dataset import WaveFakeDataset
from src.datasets.asvspoof_dataset import ASVSpoof2019DatasetOriginal
from src.datasets.WatermelonDataset import WatermelonDataset


LOGGER = logging.getLogger()


class DetectionDataset(SimpleAudioFakeDataset):
    def __init__(
        self,
        asvspoof_path=None,                             ## PATHX
        wavefake_path=None,                             ## NONE
        fakeavceleb_path=None,                          ## NONE
        asvspoof2019_path=None,                         ## NONE
        subset: str = "val",                            ## "train"
        transform=None,                                 ## NONE
        oversample: bool = True,                        ## ALL IS THE SAME
        undersample: bool = False,
        return_label: bool = True,
        reduced_number: Optional[int] = None,
        return_meta: bool = False,
    ):
        
        ## ONLY INIT THE self.<variables>
        super().__init__(
            subset=subset,              # "train"
            transform=transform,        # "None"
            return_label=return_label,  # True
            return_meta=return_meta,    # False
        )


        logging.info(f"THE ASVPROOF_PATH IS: {asvspoof_path}") 

        # THIS IS THE PROBLEM
        datasets = self._init_datasets(
            asvspoof_path=asvspoof_path, # bonafied
            wavefake_path=wavefake_path, 
            fakeavceleb_path=fakeavceleb_path,
            asvspoof2019_path=asvspoof2019_path,
            subset=subset,
        )

        ## Concat all the lists of datasets if there are many into single one
        self.samples = pd.concat([ds.samples for ds in datasets], ignore_index=True)

        if oversample:
            # GO here
            self.oversample_dataset()
        elif undersample:
            self.undersample_dataset()

        if reduced_number:
            LOGGER.info(f"Using reduced number of samples - {reduced_number}!")
            self.samples = self.samples.sample(
                min(len(self.samples), reduced_number),
                random_state=42,
            )

    def _init_datasets(
        self,
        asvspoof_path: Optional[str],
        wavefake_path: Optional[str],
        fakeavceleb_path: Optional[str],
        asvspoof2019_path: Optional[str],
        subset: str,
    ) -> List[SimpleAudioFakeDataset]:
        datasets = []

        logging.info(f"HERE ASVPROOF_PATH IS: {asvspoof_path}")
        if asvspoof_path is not None:
            # This is the problem
            #asvspoof_dataset = DeepFakeASVSpoofDataset(asvspoof_path, subset=subset)
            asvspoof_dataset = WatermelonDataset(asvspoof_path, subset=subset)
            datasets.append(asvspoof_dataset)
        logging.info(f"TEST FAKE WAVE_PATH IS: {wavefake_path}")
        if wavefake_path is not None:
            wavefake_dataset = WaveFakeDataset(wavefake_path, subset=subset)
            #wavefake_dataset = WatermelonDataset(wavefake_path, subset=subset)
            datasets.append(wavefake_dataset)

        if fakeavceleb_path is not None:
            fakeavceleb_dataset = FakeAVCelebDataset(fakeavceleb_path, subset=subset)
            datasets.append(fakeavceleb_dataset)

        if asvspoof2019_path is not None:
            la_dataset = ASVSpoof2019DatasetOriginal(
                asvspoof2019_path, fold_subset=subset
            )
            datasets.append(la_dataset)

        return datasets

    def oversample_dataset(self):


        # This is where everything is used
        samples = self.samples.groupby(by=["label"])
        bona_length = len(samples.groups["bonafide"])
        spoof_length = len(samples.groups["spoof"])

        diff_length = spoof_length - bona_length

        if diff_length < 0:
            raise NotImplementedError

        if diff_length > 0:
            bonafide = samples.get_group("bonafide").sample(diff_length, replace=True)
            self.samples = pd.concat([self.samples, bonafide], ignore_index=True)

    def undersample_dataset(self):
        samples = self.samples.groupby(by=["label"])
        bona_length = len(samples.groups["bonafide"])
        spoof_length = len(samples.groups["spoof"])

        if spoof_length < bona_length:
            raise NotImplementedError

        if spoof_length > bona_length:
            spoofs = samples.get_group("spoof").sample(bona_length, replace=True)
            self.samples = pd.concat(
                [samples.get_group("bonafide"), spoofs], ignore_index=True
            )

    def get_bonafide_only(self):
        samples = self.samples.groupby(by=["label"])
        self.samples = samples.get_group("bonafide")
        return self.samples

    def get_spoof_only(self):
        samples = self.samples.groupby(by=["label"])
        self.samples = samples.get_group("spoof")
        return self.samples

