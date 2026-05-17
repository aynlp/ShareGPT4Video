import argparse
import json
import os

from lmdeploy import pipeline, ChatTemplateConfig
from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.model.utils import rewrite_ctx
from contextlib import contextmanager
import torch

def get_image_list(video_path):
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    frames = sorted(
        f for f in os.listdir(video_path)
        if f.lower().endswith(exts) and not f.startswith(".")
    )
    img_path_list = []
    for frame in frames:
        img_path = os.path.join(video_path, frame)
        img_path_list.append(img_path)
    return img_path_list

def _forward_4khd_7b(self, images):
    """internlm-xcomposer2-4khd-7b vit forward."""
    outputs = [x.convert('RGB') for x in images]
    outputs = [self.HD_transform(x, hd_num=9) for x in outputs]
    outputs = [
        self.model.vis_processor(x).unsqueeze(0).to(dtype=torch.half)
        for x in outputs
    ]
    embeds, split = self.model.vit(outputs, self.model.plora_glb_GN,
                                    self.model.plora_sub_GN)
    embeds = self.model.vision_proj(embeds)
    embeds = torch.split(embeds, split, dim=1)
    embeds = [x.squeeze() for x in embeds]
    return embeds

@contextmanager
def custom_forward():
    origin_func_path = [
        'lmdeploy.vl.model.xcomposer2.Xcomposer2VisionModel._forward_4khd_7b',
    ]
    rewrite_func = [
        _forward_4khd_7b
    ]
    with rewrite_ctx(origin_func_path, rewrite_func):
        yield

class VideoData():
    def __init__(self, video_path, frame_interval_sec=1.0):
        self.video_path = video_path
        self.frame_interval_sec = float(frame_interval_sec)
        self.img_path_list = get_image_list(self.video_path)
        self.frame_ptr = 0
        self.caption_list = [""]	    # hack for unified code, remember to remove the first item

    @property
    def is_finished(self):
        return self.frame_ptr == len(self.img_path_list)
    
    def get_prepared_data(self,):
        curr_img = load_image(self.img_path_list[self.frame_ptr])
        t0 = 1 + self.frame_ptr * self.frame_interval_sec
        t1 = 2 + self.frame_ptr * self.frame_interval_sec
        if self.frame_ptr == 0:
            query = (
                "Write a prestige-documentary-style shot description for the opening "
                "segment labeled [{:.0f}s-{:.0f}s]. Output ONE continuous paragraph "
                "covering densely and concretely: subject action and appearance, shot "
                "type and camera movement, lighting quality and color palette, depth "
                "of field and atmosphere, mood and genre framing; use '->' arrows for "
                "causal beats where helpful. "
                "Do NOT output any time bracket prefix. Target 70-150 words."
            ).format(t0, t1)
        else:
            query = (
                "Write a prestige-documentary-style shot description for the segment "
                "labeled [{:.0f}s-{:.0f}s] (frame {} -> frame {}). Output ONE continuous "
                "paragraph covering densely and concretely: subject action and appearance "
                "(keep continuity with the prior segment), shot type and camera movement, "
                "lighting quality and color palette, depth of field and atmosphere, mood "
                "and genre framing; use '->' arrows for causal beats where helpful. "
                "Do NOT output any time bracket prefix. Do NOT restate prior-segment text. "
                "Target 70-150 words. Prior segment: {}"
            ).format(t0, t1, self.frame_ptr, self.frame_ptr + 1, self.caption_list[-1])
        self.frame_ptr += 1
        return (query, curr_img)
    
    def get_finish_data(self):
        def _fmt(t):
            return str(int(t)) if t == int(t) else f"{t:g}"
        segments = []
        for idx, caption in enumerate(self.caption_list[1:], start=1):
            t0 = 1 + (idx - 1) * self.frame_interval_sec
            t1 = 2 + (idx - 1) * self.frame_interval_sec
            segments.append(f"[{_fmt(t0)}s-{_fmt(t1)}s] {caption.strip()}")
        timeline = "\n\n".join(segments)
        return dict(
            video_path=self.video_path,
            frame_num=len(self.img_path_list),
            timeline=timeline,
        )
    
    def record_caption(self, caption):
        self.caption_list.append(caption)


class VideoPool():
    def __init__(self, pool_size=6, video_path=None, frame_interval_sec=1.0):
        self.size = pool_size
        self.frame_interval_sec = float(frame_interval_sec)
        self.video_path = json.load(open(video_path, 'r'))
        self.video_ptr = 0
        self.video_pool = []
        self._init_pool()

    def _load(self):
        # load ptr video
        video = VideoData(
            self.video_path[self.video_ptr],
            frame_interval_sec=self.frame_interval_sec,
        )
        self.video_ptr += 1
        return video

    def get_batch_data(self):
        batch_data = []
        for video in self.video_pool:
            data = video.get_prepared_data()
            batch_data.append(data)
        return batch_data

    def _init_pool(self):
        while len(self.video_pool) < self.size and \
                self.video_ptr < len(self.video_path):
            print("Load Video")
            self.video_pool.append(self._load())

    def record_caption(self, caption_list):
        # put the model generation back to the video list
        for caption, video in zip(caption_list, self.video_pool):
            video.record_caption(caption)

    def check_finished_video(self,):
        remove_list = []
        finish_list = []
        for video in self.video_pool:
            if video.is_finished:
                final_data = video.get_finish_data()
                finish_list.append(final_data)
                remove_list.append(video)
        for remove_item in remove_list:
            self.video_pool.remove(remove_item)
        self._init_pool()
        return finish_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model-name", type=str,
                        default="/mnt/petrelfs/chenlin/MLLM/hw_home/hf_ckpts/ShareGPT4Video/sharegpt4video/sharecaptioner_v1")
    parser.add_argument("--videos-file", type=str, default="describe.json",
                        help="a list, each element is a string for image path")
    parser.add_argument("--save-path", type=str, default="outputs/")
    parser.add_argument(
        "--frame-interval-sec",
        type=float,
        default=1.0,
        help="Seconds between consecutive extracted frames; must match your ffmpeg/keyframe step (e.g. fps=1 for 1s).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=("pytorch", "turbomind"),
        default="pytorch",
        help="pytorch: native HF weights (needed for ShareCaptioner PLoRA). turbomind: faster when supported.",
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    backend_cfg = (
        PytorchEngineConfig() if args.backend == "pytorch" else TurbomindEngineConfig()
    )
    model = pipeline(
        args.model_name,
        backend_config=backend_cfg,
        chat_template_config=ChatTemplateConfig(model_name='internlm-xcomposer2-4khd'),
    )
    data_pool = VideoPool(
        pool_size=args.batch_size,
        video_path=args.videos_file,
        frame_interval_sec=args.frame_interval_sec,
    )
    cnt = 0
    while True:
        batch_data = data_pool.get_batch_data()
        if len(batch_data) == 0:
            break
        with custom_forward():
            responses = model(batch_data)
        responses = [resp.text for resp in responses]
        data_pool.record_caption(responses)
        finish_list = data_pool.check_finished_video()
        cnt += 1
        for finish_data in finish_list:
            filename = finish_data['video_path'].split('/')[-1]
            with open(os.path.join(args.save_path, filename+'.json'), 'w') as f:
                f.write(json.dumps(finish_data, indent=2, ensure_ascii=False))