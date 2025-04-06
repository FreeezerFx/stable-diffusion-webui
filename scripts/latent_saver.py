from modules import scripts, processing, shared

class LatentSaver(scripts.Script):
    def title(self):
        return "Latent Saver"

    def process(self, p: processing.StableDiffusionProcessing):
        p.extra_args["store_latents"] = True

    def postprocess(self, p, processed, *args):
        if hasattr(processed, 'latents'):
            for i, latent in enumerate(processed.latents):
                path = shared.opts.outdir_samples or p.outpath_samples
                torch.save(latent.cpu(), f"{path}/latent_{processed.seed}_{i}.pt")
