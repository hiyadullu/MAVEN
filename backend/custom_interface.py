import torch
from speechbrain.inference.interfaces import Pretrained

class CustomEmotionRecognizer(Pretrained):
    def classify_file(self, wav):
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)

        probs = self.mods.classifier(self.mods.model(wav))
        score, index = torch.max(probs, dim=1)

        emotions = ["angry", "happy", "sad", "neutral"]
        return {"prediction": emotions[index.item()], "confidence": score.item()}
