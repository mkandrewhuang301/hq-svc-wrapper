# HQ-SVC (AAAI 2026) zero-shot singing voice conversion on Replicate. Private test packaging.
import os, sys, types, tempfile
import numpy as np, torch, soundfile as sf
from cog import BasePredictor, Input, Path

# HQ-SVC's code and weights are fetched from their public sources at build time (cog.yaml); this repo holds only this wrapper.
ROOT = '/hqsvc'
os.chdir(ROOT)
sys.path.append(ROOT); sys.path.append(os.path.join(ROOT, 'utils'))
# fairseq is imported by modules HQ-SVC inference never calls; stub it instead of installing it.
_fs = types.ModuleType('fairseq'); _fs.checkpoint_utils = types.SimpleNamespace()
sys.modules['fairseq'] = _fs; sys.modules['fairseq.checkpoint_utils'] = _fs.checkpoint_utils
_tl = torch.load
torch.load = lambda *a, **k: _tl(*a, **{**k, 'weights_only': k.get('weights_only', False)})


class Predictor(BasePredictor):
    def setup(self):
        from logger.utils import load_config
        from utils.models.models_v2_beta import load_hq_svc
        from utils.vocoder import Vocoder
        from utils.data_preprocessing import load_facodec, load_f0_extractor, load_volume_extractor, get_processed_file
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        A = load_config('configs/hq_svc_infer.yaml'); A.config = 'configs/hq_svc_infer.yaml'; A.device = self.dev
        self.A = A
        self.V = Vocoder(vocoder_type='nsf-hifigan', vocoder_ckpt='utils/pretrain/nsf_hifigan/model', device=self.dev)
        self.G = load_hq_svc(mode='infer', device=self.dev, model_path=A.model_path, args=A); self.G.eval()
        fe, fd = load_facodec(self.dev); f0x = load_f0_extractor(A); volx = load_volume_extractor(A)
        self.proc = lambda p: get_processed_file(p, A.sample_rate, A.encoder_sr, self.V, volx, f0x, fe, fd, None, None, device=self.dev)

    def predict(
        self,
        source_audio: Path = Input(description="Clean isolated singing to convert"),
        reference_audio: Path = Input(description="Target voice clip (speech or singing), ideally 10-60 s"),
        key_shift: int = Input(description="Semitones to shift the melody", default=0, ge=-12, le=12),
        auto_key: bool = Input(description="Shift the melody toward the reference voice's range", default=False),
    ) -> Path:
        with torch.no_grad():
            t = self.proc(str(reference_audio))
            spk, tar_f0 = t['spk'].squeeze().to(self.dev), t['f0_origin']
            s = self.proc(str(source_audio))
            shift = float(key_shift)
            if auto_key:
                sv, tv = s['f0_origin'][s['f0_origin'] > 0], tar_f0[tar_f0 > 0]
                if len(sv) and len(tv):
                    shift = round(12 * np.log2(tv.mean() / sv.mean()))
            f0 = s['f0'].unsqueeze(0).to(self.dev) * 2 ** (shift / 12)
            A = self.A
            mel = self.G(s['vq_post'].unsqueeze(0).to(self.dev), f0, s['vol'].unsqueeze(0).to(self.dev), spk,
                         gt_spec=None, infer=True, infer_speedup=A.infer_speedup, method=A.infer_method, vocoder=self.V)
            wav = self.V.infer(mel, f0)
        out = Path(tempfile.mkdtemp()) / 'converted.wav'
        sf.write(str(out), wav.squeeze().cpu().numpy(), 44100)
        return out
