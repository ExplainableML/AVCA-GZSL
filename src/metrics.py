import numpy as np
import torch

from src.utils import evaluate_dataset, get_best_evaluation, evaluate_dataset_baseline


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class AverageNonzeroTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        super(AverageNonzeroTripletsMetric, self).__init__()
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'


class PercentOverlappingClasses(Metric):
    def __init__(self):
        super(PercentOverlappingClasses, self).__init__()
        self.values = []

    def __call__(self, outputs, target, loss):
        labels1, labels2 = target
        assert len(labels1) == len(labels2)
        percent_overlap = len(torch.where(labels1.eq(labels2))[0]) / len(labels1)
        self.values.append(percent_overlap)

    def reset(self):
        self.values = []

    def value(self):
        value = np.mean(self.values)
        assert value == 0.
        return {"class_overlap": value}

    def name(self):
        return "Average p,q class overlap [%]"


class DetailedLosses(Metric):

    def __init__(self):
        super(DetailedLosses, self).__init__()
        self.cmd = []
        self.ct = []
        self.l_rec = []
        self.l_cta = []
        self.l_ctv = []
        self.l_ta = []
        self.l_at = []
        self.l_tv = []
        self.l_vt = []

    def __call__(self, outputs, target, loss):
        self.l_rec.append(loss[1]["cmd"]["l_rec"].item())
        self.l_cta.append(loss[1]["cmd"]["l_cta"].item())
        self.l_ctv.append(loss[1]["cmd"]["l_ctv"].item())
        self.l_ta.append(loss[1]["ct"]["l_ta"].item())
        self.l_at.append(loss[1]["ct"]["l_at"].item())
        self.l_tv.append(loss[1]["ct"]["l_tv"].item())
        self.l_vt.append(loss[1]["ct"]["l_vt"].item())
        self.cmd.append(self.l_rec[-1] + self.l_cta[-1] + self.l_ctv[-1])
        self.ct.append(self.l_ta[-1] + self.l_at[-1] + self.l_tv[-1] + self.l_vt[-1])

    def reset(self):
        self.cmd = []
        self.ct = []
        self.l_rec = []
        self.l_cta = []
        self.l_ctv = []
        self.l_ta = []
        self.l_at = []
        self.l_tv = []
        self.l_vt = []

    def value(self):
        return {
            "cmd": np.mean(self.cmd),
            "ct": np.mean(self.ct),
            "l_rec": np.mean(self.l_rec),
            "l_cta": np.mean(self.l_cta),
            "l_ctv": np.mean(self.l_ctv),
            "l_ta": np.mean(self.l_ta),
            "l_at": np.mean(self.l_at),
            "l_tv": np.mean(self.l_tv),
            "l_vt": np.mean(self.l_vt)
        }

    def name(self):
        return "Debug losses"


class TargetDifficulty(Metric):
    def __init__(self, margin, distance_fn):
        super(TargetDifficulty, self).__init__()
        self.margin = margin
        self.distance_fn = distance_fn
        self.easy_audio = []
        self.hard_audio = []
        self.semi_hard_audio = []
        self.easy_video = []
        self.hard_video = []
        self.semi_hard_video = []

    def __call__(self, outputs, target, loss):
        # model output is:
        # x_t1, a1, v1, t1, a2, v2, t2, x_ta1, x_tv1, x_tt1, x_ta2, x_tv2
        _, a1, v1, t1, a2, v2, _, _, _, _, _, _ = outputs
        easy_audio, hard_audio, semi_hard_audio = self._get_triplet_difficulty(anchor=t1, positive=a1, negative=a2,
                                                                               margin=self.margin)
        easy_video, hard_video, semi_hard_video = self._get_triplet_difficulty(anchor=t1, positive=v1, negative=v2,
                                                                               margin=self.margin)
        self.easy_audio.append(easy_audio)
        self.hard_audio.append(hard_audio)
        self.semi_hard_audio.append(semi_hard_audio)
        self.easy_video.append(easy_video)
        self.hard_video.append(hard_video)
        self.semi_hard_video.append(semi_hard_video)

    def reset(self):
        self.easy_audio = []
        self.hard_audio = []
        self.semi_hard_audio = []
        self.easy_video = []
        self.hard_video = []
        self.semi_hard_video = []

    def value(self):
        return {
            "easy_audio": np.mean(self.easy_audio),
            "hard_audio": np.mean(self.hard_audio),
            "semi_hard_audio": np.mean(self.semi_hard_audio),
            "easy_video": np.mean(self.easy_video),
            "hard_video": np.mean(self.hard_video),
            "semi_hard_video": np.mean(self.semi_hard_video),
        }

    def name(self):
        return "Target difficulties"

    def _get_triplet_difficulty(self, anchor, positive, negative, margin):
        distance_positive = self.distance_fn(anchor, positive)
        distance_negative = self.distance_fn(anchor, negative)
        easy_targets = distance_negative > distance_positive + margin
        hard_targets = distance_negative < distance_positive
        semi_hard_targets = distance_negative < distance_positive + margin
        return (
            np.mean(easy_targets.cpu().numpy()),
            np.mean(hard_targets.cpu().numpy()),
            np.mean(semi_hard_targets.cpu().numpy())
        )


class MeanClassAccuracy(Metric):
    def __init__(self, model, dataset, device, distance_fn, new_model_attention=False,model_devise=False,apn=False,args=None):
        super(MeanClassAccuracy, self).__init__()
        self.model = model
        self.model_devise=model_devise
        self.new_model_attention=new_model_attention
        self.dataset = dataset
        self.device = device
        self.apn=apn
        self.distance_fn = distance_fn
        self.args = args
        self.audio_seen = []
        self.audio_unseen = []
        self.audio_hm = []
        self.audio_recall = []
        self.audio_beta = []
        self.audio_zsl=[]

        self.video_seen = []
        self.video_unseen = []
        self.video_hm = []
        self.video_recall = []
        self.video_beta = []
        self.video_zsl=[]


        self.both_seen = []
        self.both_unseen = []
        self.both_hm = []
        self.both_recall = []
        self.both_beta = []
        self.both_zsl=[]

    def __call__(self, outputs, target, loss_outputs):

        if self.new_model_attention==False and self.model_devise==False and self.apn==False:
            evaluation = evaluate_dataset(dataset=self.dataset, model=self.model, device=self.device,
                                          distance_fn=self.distance_fn, args=self.args)
        else:
            evaluation = evaluate_dataset_baseline(dataset=self.dataset, model=self.model, device=self.device,
                                                   distance_fn=self.distance_fn,
                                                   new_model_attention=self.new_model_attention,
                                                   model_devise=self.model_devise,
                                                   apn=self.apn,
                                                   args=self.args)

        self.audio_seen.append(evaluation["audio"]["seen"])
        self.audio_unseen.append(evaluation["audio"]["unseen"])
        self.audio_hm.append(evaluation["audio"]["hm"])
        self.audio_recall.append(evaluation["audio"]["recall"])
        self.audio_beta.append(evaluation["audio"]["beta"])
        self.audio_zsl.append(evaluation["audio"]["zsl"])

        self.video_seen.append(evaluation["video"]["seen"])
        self.video_unseen.append(evaluation["video"]["unseen"])
        self.video_hm.append(evaluation["video"]["hm"])
        self.video_recall.append(evaluation["video"]["recall"])
        self.video_beta.append(evaluation["video"]["beta"])
        self.video_zsl.append(evaluation["video"]["zsl"])

        self.both_seen.append(evaluation["both"]["seen"])
        self.both_unseen.append(evaluation["both"]["unseen"])
        self.both_hm.append(evaluation["both"]["hm"])
        self.both_recall.append(evaluation["both"]["recall"])
        self.both_beta.append(evaluation["both"]["beta"])
        self.both_zsl.append(evaluation["both"]["zsl"])

    def reset(self):
        self.audio_seen = []
        self.audio_unseen = []
        self.audio_hm = []
        self.audio_recall = []
        self.audio_beta = []
        self.audio_zsl=[]

        self.video_seen = []
        self.video_unseen = []
        self.video_hm = []
        self.video_recall = []
        self.video_beta = []
        self.video_zsl=[]

        self.both_seen = []
        self.both_unseen = []
        self.both_hm = []
        self.both_recall = []
        self.both_beta = []
        self.both_zsl=[]

    def value(self):
        return {
            "audio_seen": np.mean(self.audio_seen),
            "audio_unseen": np.mean(self.audio_unseen),
            "audio_hm": np.mean(self.audio_hm),
            "audio_recall": np.mean(self.audio_recall, axis=0),
            "audio_beta": np.mean(self.audio_beta),
            "audio_zsl":np.mean(self.audio_zsl),

            "video_seen": np.mean(self.video_seen),
            "video_unseen": np.mean(self.video_unseen),
            "video_hm": np.mean(self.video_hm),
            "video_recall": np.mean(self.video_recall, axis=0),
            "video_beta": np.mean(self.video_beta),
            "video_zsl":np.mean(self.video_zsl),

            "both_seen": np.mean(self.both_seen),
            "both_unseen": np.mean(self.both_unseen),
            "both_hm": np.mean(self.both_hm),
            "both_recall": np.mean(self.both_recall, axis=0),
            "both_beta": np.mean(self.both_beta),
            "both_zsl":np.mean(self.both_zsl)
        }

    def name(self):
        return "Mean class accuracies per modality"
