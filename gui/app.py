"""Simple local GUI for DJtransGAN inference: drag & drop two tracks, pick
cue points, hit process, preview the generated transition.

Run with:
    python gui/app.py [--n_gpu 0] [--share]
"""
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gradio as gr

from djtransgan.config  import settings
from djtransgan.utils   import download_pretrained
from djtransgan.utils   import check_exist, squeeze_dim, get_filename, load_pt
from djtransgan.utils   import load_audio, out_audio
from djtransgan.model   import get_generator
from djtransgan.process import preprocess, postprocess

torch.manual_seed(settings.RANDOM_SEED)

MODELS = {
    'minmax (standaard)': './pretrained/djtransgan_minmax.pt',
    'least squares'      : './pretrained/djtransgan_least_square.pt',
}

OUT_DIR = os.path.join(settings.STORE_DIR, 'gui')

_generator_cache = {}


def list_devices():
    devices = ['cpu']
    if torch.cuda.is_available():
        devices += [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    return devices


def get_cached_generator(g_path):
    if g_path not in _generator_cache:
        generator = get_generator()
        if os.path.exists(g_path):
            generator.load_state_dict(load_pt(g_path))
        generator.eval()
        _generator_cache[g_path] = generator
    return _generator_cache[g_path]


def run_inference(prev_track, next_track, prev_cue, next_cue, model_choice, device_choice):
    if prev_track is None or next_track is None:
        return None, None, 'Upload eerst twee tracks.'

    try:
        device = torch.device(device_choice)
        g_path = MODELS[model_choice]

        generator = get_cached_generator(g_path).to(device)

        prev_audio = load_audio(prev_track)
        next_audio = load_audio(next_track)

        (pair_audio, timestamps), (pair_audio_for_g, cue_for_g) = preprocess(
            prev_audio, next_audio, float(prev_cue), float(next_cue)
        )
        pair_audio_for_g = [audio.to(device) for audio in pair_audio_for_g]
        cue_for_g = cue_for_g.to(device)

        mix_audio, _ = generator.infer(*pair_audio_for_g, cue_region=cue_for_g)
        mix_audio = mix_audio.to('cpu')
        post_mix_audio, _ = postprocess(mix_audio, pair_audio, timestamps, cue_for_g.to('cpu'))

        saved_id = f'{get_filename(prev_track)}_{get_filename(next_track)}'
        short_path = os.path.join(OUT_DIR, f'{saved_id}_short.wav')
        full_path = os.path.join(OUT_DIR, f'{saved_id}_full.wav')
        check_exist(short_path)
        check_exist(full_path)
        out_audio(squeeze_dim(mix_audio).to(torch.float32), short_path)
        out_audio(squeeze_dim(post_mix_audio).to(torch.float32), full_path)

        return short_path, full_path, f'Klaar (device: {device}).'
    except Exception as e:
        return None, None, f'Fout: {e}'


def build_ui(default_device):
    with gr.Blocks(title='DJtransGAN') as demo:
        gr.Markdown('# DJtransGAN — DJ-transitie genereren')
        gr.Markdown(
            'Sleep twee tracks naar binnen, geef het cue-point van elke track op '
            '(in seconden: het punt waar de vorige track helemaal is weg-/ingefade), '
            'en klik op Genereer.'
        )

        with gr.Row():
            with gr.Column():
                prev_track = gr.Audio(label='Vorige track', type='filepath')
                prev_cue = gr.Number(label='Cue-point vorige track (seconden)', value=0)
            with gr.Column():
                next_track = gr.Audio(label='Volgende track', type='filepath')
                next_cue = gr.Number(label='Cue-point volgende track (seconden)', value=0)

        with gr.Row():
            model_choice = gr.Dropdown(choices=list(MODELS.keys()), value=list(MODELS.keys())[0], label='Model')
            device_choice = gr.Dropdown(choices=list_devices(), value=default_device, label='Device')

        run_button = gr.Button('Genereer transitie', variant='primary')
        status = gr.Textbox(label='Status', interactive=False)

        with gr.Row():
            short_out = gr.Audio(label='Transitie (kort)')
            full_out = gr.Audio(label='Volledige mix')

        run_button.click(
            fn=run_inference,
            inputs=[prev_track, next_track, prev_cue, next_cue, model_choice, device_choice],
            outputs=[short_out, full_out, status],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description='DJtransGAN GUI')
    parser.add_argument('--n_gpu', default=-1, help='-1 voor cpu, anders cuda device index (bv. 0)')
    parser.add_argument('--share', action='store_true', help='maak een publieke (tijdelijke) gradio-link')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()

    print('Pretrained modellen downloaden (indien nog niet aanwezig) ...')
    download_pretrained()

    default_device = 'cpu' if int(args.n_gpu) == -1 else f'cuda:{args.n_gpu}'
    if default_device not in list_devices():
        default_device = 'cpu'

    demo = build_ui(default_device)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == '__main__':
    main()
