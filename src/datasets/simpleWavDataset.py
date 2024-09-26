import os
import torchaudio
from torch.utils.data import Dataset

class SimpleWavDataset(Dataset):
    def __init__(self, bonafide_dir, fake_dir, transform=None):
        """
        Args:
            bonafide_dir (str): Directory containing bonafide (real) audio files.
            fake_dir (str): Directory containing fake audio files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.bonafide_dir = bonafide_dir
        self.fake_dir = fake_dir
        self.transform = transform

        # Load bonafide files and label them as 0 (bonafide)
        self.bonafide_files = [(os.path.join(self.bonafide_dir, file), 0) 
                               for file in os.listdir(self.bonafide_dir) 
                               if file.endswith('.wav')]

        # Load fake files and label them as 1 (fake)
        self.fake_files = [(os.path.join(self.fake_dir, file), 1) 
                           for file in os.listdir(self.fake_dir) 
                           if file.endswith('.wav')]

        # Combine the bonafide and fake files into one dataset
        self.samples = self.bonafide_files + self.fake_files

    def __len__(self):
        """Returns the total number of samples"""
        return len(self.samples)

    def __getitem__(self, idx):
        """Retrieve a sample by index"""
        wav_path, label = self.samples[idx]

        # Load the wav file
        waveform, sample_rate = torchaudio.load(wav_path)

        # Apply any transformations (e.g., LFCC, MFCC)
        if self.transform:
            waveform = self.transform(waveform)

        # Return the waveform and its corresponding label (0 for bonafide, 1 for fake)
        return waveform, label
