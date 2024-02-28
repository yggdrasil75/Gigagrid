import os
import numpy as np
import torch

from modules import scripts
from modules.processing import create_infotext
from modules.sd_samplers_kdiffusion import KDiffusionSampler
try:
	from modules.sd_samplers_timesteps import CompVisSampler as TimestepsSampler, samplers_timesteps
except:
	# Fallback if webui is an old version without sd_samplers_timesteps
	class TimestepsSampler:
		callback_state = None
	from modules.sd_samplers_compvis import samplers_data_compvis as samplers_timesteps
from modules.sd_samplers_common import samples_to_images_tensor
from modules.images import save_image
from gigacore import quickTimeMath
from PIL import Image
import threading

CallbackStateKDiff:callable = KDiffusionSampler.callback_state
CallbackStateTimestep = TimestepsSampler.callback_state

def SamplerType(Sampler):
	for sampler in samplers_timesteps:
		if sampler[0] == Sampler:
			return True
	return False

def singleIMG(sample, index, approximation=None):
	x_sample = sample[index]
	x_sample = samples_to_images_tensor(x_sample.unsqueeze(0), approximation)[0] * 0.5 + 0.5

	#x_sample = torch.stack(x_sample).float()
	x_sample = torch.clamp((x_sample + 1.0), min=0.0, max=None)
	x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
	x_sample = x_sample.astype(np.uint8)

	return Image.fromarray(x_sample)

def getCurrentStepSavepath(p, step):
	# Get the index of the current step in multistep
	stepmult = p.giga['multistep'].index(step)

	# Calculate the step increment based on the number of values in multistep
	num_multistep = len(p.giga['multistep'])

	# Calculate the indices in savePath based on the current step
	save_path_indices = [stepmult + i * num_multistep for i in range(0, len(p.gigauncomp['savePath']) // num_multistep)]

	# Get the corresponding save paths
	save_paths = [p.gigauncomp['savePath'][index] for index in save_path_indices]

	return save_paths

def Save(p, img, path, index):
	def _save(p, img, path, index):


		print(f"saving to {path}")
		result = save_image(img, path=os.path.dirname(path), basename="",
			forced_filename=os.path.basename(path), save_to_dirs=False,
			extension=p.giga['format'], p=p, prompt=p.prompt[index],seed=p.seed, 
			info=create_infotext(p, p.prompt, p.seed,
						p.subseed, f"Time taken: {quickTimeMath(p.giga['start'])}", p.iteration, index))
		return result
	return threading.Thread(_save(p, img, path, index)).start()

class Script(scripts.Script):
	def title(self):
		return "Aid script for GigaGrid"

	def show(self, is_img2img):
		return scripts.AlwaysVisible

	def process(self, p):
		if hasattr(p, 'giga'):


			def callback_state(self, d):
				#print('testcallback')
				"""
				callback_state runs after each processing step
				"""
				if SamplerType(p.sampler_name):
					orig_callback_state = CallbackStateTimestep
				else:
					orig_callback_state = CallbackStateKDiff

				current_step = d["i"]

				hr = True if getattr(p, "enable_hr", False) else False

				abs_step = current_step
				if p.sampler_name == 'ddim':
					abs_step += 1
				hr_active = False
				if hr:
					hr_active = True if getattr(p, "enable_hr", False) else False
					if hr_active:
						abs_step = current_step + p.steps
						if not hasattr(p, 'intermed_hires_start'):
							p.intermed_hires_start = abs_step

				if abs_step in p.giga['multistep']:
					savePaths = getCurrentStepSavepath(p, abs_step)
					for index in range(0, p.batch_size):

						path = savePaths[index]
						base_name, _ = os.path.splitext(path)

						img = singleIMG(d['denoised'],index, 3) #x is blobs. denoised is not. 
						base_name = os.path.basename(base_name)

						# Don't continue with no image (can happen with live preview subject setting)
						if img is not None:
							# Don't save first step or if before start_at
							if not (abs_step == 0):
								# generate png-info
								infotext = create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments=[], position_in_batch=index % p.batch_size, iteration=index // p.batch_size)
								infotext = f'{infotext}, intermediate: {abs_step:03d}'

								if ('inf_grid_out_width' in p.gigauncomp.keys() and abs_step in p.gigauncomp['inf_grid_out_width'].keys()) and ('simpleUpscaleH' in p.gigauncomp.keys() and abs_step in p.gigauncomp['simpleUpscaleH'].keys()):
									width = p.gigauncomp['inf_grid_out_width'][abs_step]
									height = p.gigauncomp['simpleUpscaleH'][abs_step]
								elif ('inf_grid_out_width' in p.gigauncomp.keys() and abs_step in p.gigauncomp['inf_grid_out_width'].keys()):
									height = img.width / img.height * p.gigauncomp['inf_grid_out_width'][abs_step]
								elif ('simpleUpscaleH' in p.gigauncomp.keys() and abs_step in p.gigauncomp['simpleUpscaleH'].keys()):
									width = img.height / img.width * p.gigauncomp['simpleUpscaleH'][abs_step]
								else:
									height = img.height
									width = img.width

								img = img.resize((width, height), resample=img.LANCZOS)

								# save intermediate image
								Save(p=p, img=img, path=path, index=index)
				return orig_callback_state(self, d)

			setattr(KDiffusionSampler, "callback_state", callback_state)
			setattr(TimestepsSampler, "callback_state", callback_state)

	def postprocess(self, p, processed):
		setattr(KDiffusionSampler, "callback_state", CallbackStateKDiff)
		setattr(TimestepsSampler, "callback_state", CallbackStateTimestep)