##################
# Stable Diffusion Infinity Grid Generator
#
# Author: Alex 'mcmonkey' Goodwin
# GitHub URL: https://github.com/mcmonkeyprojects/sd-infinity-grid-generator-script
# Created: 2022/12/08
# Last updated: 2023/02/19
# License: MIT
#
# For usage help, view the README.md file in the extension root, or via the GitHub page.
#
##################

import gradio as gr
import os
import numpy
import ctypes
from copy import copy
from datetime import datetime
from modules import images, shared, sd_models, sd_vae, sd_samplers, scripts, processing, ui_components
from modules.processing import StableDiffusionProcessing, process_images, Processed
from modules.shared import opts
from PIL import Image
import gridgencore as core
from gridgencore import cleanName, getBestInList, chooseBetterFileName, GridSettingMode, fixNum, applyField, registerMode
import json
import torch

######################### Constants #########################
refresh_symbol = '\U0001f504'  # 🔄
fill_values_symbol = "\U0001f4d2"  # 📒
INF_GRID_README = "https://github.com/mcmonkeyprojects/sd-infinity-grid-generator-script"
core.EXTRA_FOOTER = 'Images area auto-generated by an AI (Stable Diffusion) and so may not have been reviewed by the page author before publishing.\n<script src="a1111webui.js?vary=9"></script>'
core.EXTRA_ASSETS = ["a1111webui.js"]

######################### Value Modes #########################

def getModelFor(name):
	return getBestInList(name, map(lambda m: m.title, sd_models.checkpoints_list.values()))

def applyModel(p, v):
	opts.sd_model_checkpoint = getModelFor(v)
	sd_models.reload_model_weights()

def cleanModel(p, v):
	actualModel = getModelFor(v)
	if actualModel is None:
		raise RuntimeError(f"Invalid parameter '{p}' as '{v}': model name unrecognized - valid {list(map(lambda m: m.title, sd_models.checkpoints_list.values()))}")
	return chooseBetterFileName(v, actualModel)

def getVaeFor(name):
	return getBestInList(name, sd_vae.vae_dict.keys())

def applyVae(p, v):
	vaeName = cleanName(v)
	if vaeName == "none":
		vaeName = "None"
	elif vaeName in ["auto", "automatic"]:
		vaeName = "Automatic"
	else:
		vaeName = getVaeFor(vaeName)
	opts.sd_vae = vaeName
	sd_vae.reload_vae_weights(None)

def cleanVae(p, v):
	vaeName = cleanName(v)
	if vaeName in ["none", "auto", "automatic"]:
		return vaeName
	actualVae = getVaeFor(vaeName)
	if actualVae is None:
		raise RuntimeError(f"Invalid parameter '{p}' as '{v}': VAE name unrecognized - valid: {list(sd_vae.vae_dict.keys())}")
	return chooseBetterFileName(v, actualVae)

def applyClipSkip(p, v):
	opts.CLIP_stop_at_last_layers = int(v)

def applyCodeformerWeight(p, v):
	opts.code_former_weight = float(v)

def applyRestoreFaces(p, v):
	input = str(v).lower().strip()
	if input == "false":
		p.restore_faces = False
		return
	p.restore_faces = True
	restorer = getBestInList(input, map(lambda m: m.name(), shared.face_restorers))
	if restorer is not None:
		opts.face_restoration_model = restorer

def applyPromptReplace(p, v):
	val = v.split('=', maxsplit=1)
	if len(val) != 2:
		raise RuntimeError(f"Invalid prompt replace, missing '=' symbol, for '{v}'")
	match = val[0].strip()
	replace = val[1].strip()
	if Script.VALIDATE_REPLACE:
		if match not in p.prompt and match not in p.negative_prompt:
			raise RuntimeError(f"Invalid prompt replace, '{match}' is not in prompt '{p.prompt}' nor negative prompt '{p.negative_prompt}'")
	p.prompt = p.prompt.replace(match, replace)
	##p.negative_prompt = p.negative_prompt.replace(match, replace)
	
def applyNegPromptReplace(p, v):
	val = v.split('=', maxsplit=1)
	if len(val) != 2:
		raise RuntimeError(f"Invalid prompt replace, missing '=' symbol, for '{v}'")
	match = val[0].strip()
	replace = val[1].strip()
	if Script.VALIDATE_REPLACE:
		if match not in p.prompt and match not in p.negative_prompt:
			raise RuntimeError(f"Invalid prompt replace, '{match}' is not in prompt '{p.prompt}' nor negative prompt '{p.negative_prompt}'")
	##p.prompt = p.prompt.replace(match, replace)
	p.negative_prompt = p.negative_prompt.replace(match, replace)

def applyEnsd(p, v):
	opts.eta_noise_seed_delta = int(v)

def applyEnableHr(p, v):
	p.enable_hr = v
	if v:
		if p.denoising_strength is None:
			p.denoising_strength = 0.75
		
SEMVER_TO_ARCH = {
	(1, 0): 'tesla',
	(1, 1): 'tesla',
	(1, 2): 'tesla',
	(1, 3): 'tesla',

	(2, 0): 'fermi',
	(2, 1): 'fermi',

	(3, 0): 'kepler',
	(3, 2): 'kepler',
	(3, 5): 'kepler',
	(3, 7): 'kepler',

	(5, 0): 'maxwell',
	(5, 2): 'maxwell',
	(5, 3): 'maxwell',

	(6, 0): 'pascal',
	(6, 1): 'pascal',
	(6, 2): 'pascal',

	(7, 0): 'volta',
	(7, 2): 'volta',

	(7, 5): 'turing',

	(8, 0): 'ampere',
	(8, 6): 'ampere',
}

def getarch() -> str:
	libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
	for libname in libnames:
		try:
			cuda = ctypes.CDLL(libname)
		except OSError:
			continue
		else:
			break
	else:
		return "unknown"
	cc_major = ctypes.c_int()
	cc_minor = ctypes.c_int()

	return SEMVER_TO_ARCH.get((cc_major.value, cc_minor.value), 'unknown')
		

def setbatch(p, v):
	# Load the model information from the JSON file
	# Get the absolute path of the directory in which the script is running
	dir_path = os.path.abspath(os.path.dirname(__file__))
	#print(dir_path)
	# Load the JSON file from the current directory
	with open(os.path.join(dir_path, 'models.json')) as f:
		models_data = json.load(f)

	# Get the GPU information
	gpu = torch.cuda.get_device_name(torch.cuda.current_device)
	#print(gpu)
	architecture = getarch()
	#print(architecture)
	ram = torch.cuda.get_device_properties(torch.cuda.current_device).total_memory / 1024**3  # Convert to GB

	if v == 'bytype':
		# Search the models list for a matching brand and architecture
		for model in models_data['models']:
			if model['brand'].lower() == gpu.split()[0].lower():
				for arch in model['architectures']:
					if architecture in arch:
						# Check the RAM ranges for the current architecture
						for ram_range in arch[gpu.split()[1].lower()]:
							if 'minram' in ram_range and ram < ram_range['minram']:
								continue
							if 'maxram' in ram_range and ram > ram_range['maxram']:
								continue
							
							# Set the batch size based on the RAM range for the architecture
							batch_size = ram_range['size']
							break
						else:
							# No matching RAM range found, default to batch size of 1
							batch_size = 1
						break
				else:
					# No matching architecture found, default to batch size of 1
					batch_size = 1
				break
		else:
			# No matching brand found, default to batch size of 1
			batch_size = 1

	elif v.startswith('byid'):
		# Parse the byid values from the input string
		id_values = {}
		for value in v.split('|')[1:]:
			id, size = value.split('=')
			id_values[int(id)] = int(size)
		
		# Set the batch size based on the GPU ID
		gpu_id = torch.cuda.current_device()
		if gpu_id in id_values:
			p.batch_size = id_values[gpu_id]
		else:
			# No matching GPU ID found, default to batch size of 1
			p.batch_size = 1
	elif isinstance(v, int) or (isinstance(v, str) and v.isdigit()):
		if isinstance(v,str): v = int(v)
		p.batch_size = v
	else:
		# Invalid input value, default to batch size of 1
		p.batch_size = 1


######################### Addons #########################
hasInited = False

def tryInit():
	global hasInited
	if hasInited:
		return
	hasInited = True
	core.getModelFor = getModelFor
	core.gridCallInitHook = a1111GridCallInitHook
	core.gridCallParamAddHook = a1111GridCallParamAddHook
	core.gridCallApplyHook = a1111GridCallApplyHook
	core.gridRunnerPreRunHook = a1111GridRunnerPreRunHook
	core.gridRunnerPreDryHook = a1111GridRunnerPreDryHook
	core.gridRunnerRunPostDryHook = a1111GridRunnerPostDryHook
	core.webDataGetBaseParamData = a1111WebDataGetBaseParamData
	registerMode("Model", GridSettingMode(dry=False, type="text", apply=applyModel, clean=cleanModel, valid_list=lambda: list(map(lambda m: m.title, sd_models.checkpoints_list.values()))))
	registerMode("VAE", GridSettingMode(dry=False, type="text", apply=applyVae, clean=cleanVae, valid_list=lambda: list(sd_vae.vae_dict.keys()) + ['none', 'auto', 'automatic']))
	registerMode("Sampler", GridSettingMode(dry=True, type="text", apply=applyField("sampler_name"), valid_list=lambda: list(sd_samplers.all_samplers_map.keys())))
	registerMode("ClipSkip", GridSettingMode(dry=False, type="integer", min=1, max=12, apply=applyClipSkip))
	registerMode("Restore Faces", GridSettingMode(dry=True, type="text", apply=applyRestoreFaces, valid_list=lambda: list(map(lambda m: m.name(), shared.face_restorers)) + ["true", "false"]))
	registerMode("CodeFormer Weight", GridSettingMode(dry=True, type="decimal", min=0, max=1, apply=applyCodeformerWeight))
	registerMode("ETA Noise Seed Delta", GridSettingMode(dry=True, type="integer", apply=applyEnsd))
	registerMode("Enable HighRes Fix", GridSettingMode(dry=True, type="boolean", apply=applyEnableHr))
	registerMode("Prompt Replace", GridSettingMode(dry=True, type="text", apply=applyPromptReplace))
	registerMode("Negative Prompt Replace", GridSettingMode(dry=True, type="text", apply=applyNegPromptReplace))
	registerMode("N Prompt Replace", GridSettingMode(dry=True, type="text", apply=applyNegPromptReplace))
	for i in range(0, 10):
		registerMode(f"Prompt Replace{i}", GridSettingMode(dry=True, type="text", apply=applyPromptReplace))
		registerMode(f"Negative Prompt Replace{i}", GridSettingMode(dry=True, type="text", apply=applyNegPromptReplace))
		registerMode(f"N Prompt Replace{i}", GridSettingMode(dry=True, type="text", apply=applyNegPromptReplace))
	modes = ["var seed","seed", "width", "height"]
	fields = ["subseed", "seed",  "width", "height"]
	for field, mode in enumerate(modes):
		registerMode(mode, GridSettingMode(dry=True, type="integer", apply=applyField(fields[field])))
	modes = ["prompt", "negative prompt", "random"]
	fields = ["prompt", "negative_prompt", "randomtime"]
	for field, mode in enumerate(modes):
		registerMode(mode, GridSettingMode(dry=True, type="text", apply=applyField(fields[field])))
	
	registerMode("batch size", GridSettingMode(dry=True, type="text", apply=setbatch))
	registerMode("Steps", GridSettingMode(dry=True, type="integer", min=0, max=200, apply=applyField("steps")))
	registerMode("CFG Scale", GridSettingMode(dry=True, type="decimal", min=0, max=500, apply=applyField("cfg_scale")))
	registerMode("Tiling", GridSettingMode(dry=True, type="boolean", apply=applyField("tiling")))
	registerMode("Var Strength", GridSettingMode(dry=True, type="decimal", min=0, max=1, apply=applyField("subseed_strength")))
	registerMode("Denoising", GridSettingMode(dry=True, type="decimal", min=0, max=1, apply=applyField("denoising_strength")))
	registerMode("ETA", GridSettingMode(dry=True, type="decimal", min=0, max=1, apply=applyField("eta")))
	registerMode("Sigma Churn", GridSettingMode(dry=True, type="decimal", min=0, max=1, apply=applyField("s_churn")))
	registerMode("Sigma TMin", GridSettingMode(dry=True, type="decimal", min=0, max=1, apply=applyField("s_tmin")))
	registerMode("Sigma TMax", GridSettingMode(dry=True, type="decimal", min=0, max=1, apply=applyField("s_tmax")))
	registerMode("Sigma Noise", GridSettingMode(dry=True, type="decimal", min=0, max=1, apply=applyField("s_noise")))
	registerMode("Out Width", GridSettingMode(dry=True, type="integer", min=0, apply=applyField("inf_grid_out_width")))
	registerMode("Out Height", GridSettingMode(dry=True, type="integer", min=0, apply=applyField("inf_grid_out_height")))
	registerMode("Image Mask Weight", GridSettingMode(dry=True, type="decimal", min=0, max=1, apply=applyField("inpainting_mask_weight")))
	registerMode("HighRes Scale", GridSettingMode(dry=True, type="decimal", min=1, max=16, apply=applyField("hr_scale")))
	registerMode("HighRes Steps", GridSettingMode(dry=True, type="integer", min=0, max=200, apply=applyField("hr_second_pass_steps")))
	registerMode("HighRes Resize Width", GridSettingMode(dry=True, type="integer", apply=applyField("hr_resize_x")))
	registerMode("HighRes Resize Height", GridSettingMode(dry=True, type="integer", apply=applyField("hr_resize_y")))
	registerMode("HighRes Upscale to Width", GridSettingMode(dry=True, type="integer", apply=applyField("hr_upscale_to_x")))
	registerMode("HighRes Upscale to Height", GridSettingMode(dry=True, type="integer", apply=applyField("hr_upscale_to_y")))
	registerMode("HighRes Upscaler", GridSettingMode(dry=True, type="text", apply=applyField("hr_upscaler"), valid_list=lambda: list(map(lambda u: u.name, shared.sd_upscalers)) + list(shared.latent_upscale_modes.keys())))
	registerMode("Image CFG Scale", GridSettingMode(dry=True, type="decimal", min=0, max=500, apply=applyField("image_cfg_scale")))
	registerMode("Use Result Index", GridSettingMode(dry=True, type="integer", min=0, max=500, apply=applyField("inf_grid_use_result_index")))
	try:
		scriptList = [x for x in scripts.scripts_data if x.script_class.__module__ == "dynamic_thresholding.py"][:1]
		if len(scriptList) == 1:
			dynamic_thresholding = scriptList[0].module
			registerMode("[DynamicThreshold] Enable", GridSettingMode(dry=True, type="boolean", apply=applyField("dynthres_enabled")))
			registerMode("[DynamicThreshold] Mimic Scale", GridSettingMode(dry=True, type="decimal", min=0, max=500, apply=applyField("dynthres_mimic_scale")))
			registerMode("[DynamicThreshold] Threshold Percentile", GridSettingMode(dry=True, type="decimal", min=0.0, max=100.0, apply=applyField("dynthres_threshold_percentile")))
			registerMode("[DynamicThreshold] Mimic Mode", GridSettingMode(dry=True, type="text", apply=applyField("dynthres_mimic_mode"), valid_list=lambda: list(dynamic_thresholding.VALID_MODES)))
			registerMode("[DynamicThreshold] CFG Mode", GridSettingMode(dry=True, type="text", apply=applyField("dynthres_cfg_mode"), valid_list=lambda: list(dynamic_thresholding.VALID_MODES)))
			registerMode("[DynamicThreshold] Mimic Scale Minimum", GridSettingMode(dry=True, type="decimal", min=0.0, max=100.0, apply=applyField("dynthres_mimic_scale_min")))
			registerMode("[DynamicThreshold] CFG Scale Minimum", GridSettingMode(dry=True, type="decimal", min=0.0, max=100.0, apply=applyField("dynthres_cfg_scale_min")))
			registerMode("[DynamicThreshold] Experiment Mode", GridSettingMode(dry=True, type="integer", min=0, max=100, apply=applyField("dynthres_experiment_mode")))
			registerMode("[DynamicThreshold] Power Value", GridSettingMode(dry=True, type="decimal", min=0, max=100, apply=applyField("dynthres_power_val")))
		scriptList = [x for x in scripts.scripts_data if x.script_class.__module__ == "controlnet.py"][:1]
		if len(scriptList) == 1:
			# Hacky but works
			preprocessors_list = scriptList[0].script_class().preprocessor.keys()
			module = scriptList[0].module
			def validateParam(p, v):
				if not shared.opts.data.get("control_net_allow_script_control", False):
					raise RuntimeError("ControlNet options cannot currently work, you must enable 'Allow other script to control this extension' in Settings -> ControlNet first")
				return v
			registerMode("[ControlNet] Enable", GridSettingMode(dry=True, type="boolean", apply=applyField("control_net_enabled"), clean=validateParam))
			registerMode("[ControlNet] Preprocessor", GridSettingMode(dry=True, type="text", apply=applyField("control_net_module"), clean=validateParam, valid_list=lambda: list(preprocessors_list)))
			registerMode("[ControlNet] Model", GridSettingMode(dry=True, type="text", apply=applyField("control_net_model"), clean=validateParam, valid_list=lambda: list(list(module.cn_models.keys()))))
			registerMode("[ControlNet] Weight", GridSettingMode(dry=True, type="decimal", min=0.0, max=2.0, apply=applyField("control_net_weight"), clean=validateParam))
			registerMode("[ControlNet] Guidance Strength", GridSettingMode(dry=True, type="decimal", min=0.0, max=1.0, apply=applyField("control_net_guidance_strength"), clean=validateParam))
			registerMode("[ControlNet] Annotator Resolution", GridSettingMode(dry=True, type="integer", min=0, max=2048, apply=applyField("control_net_pres"), clean=validateParam))
			registerMode("[ControlNet] Threshold A", GridSettingMode(dry=True, type="integer", min=0, max=256, apply=applyField("control_net_pthr_a"), clean=validateParam))
			registerMode("[ControlNet] Threshold B", GridSettingMode(dry=True, type="integer", min=0, max=256, apply=applyField("control_net_pthr_b"), clean=validateParam))
			registerMode("[ControlNet] Image", GridSettingMode(dry=True, type="text", apply=core.applyFieldAsImageData("control_net_input_image"), clean=validateParam, valid_list=lambda: core.listImageFiles()))
	except ModuleNotFoundError as e:
		print(f"Infinity Grid Generator failed to import a dependency module: {e}")
		pass

######################### Actual Execution Logic #########################

def a1111GridCallInitHook(gridCall):
	gridCall.replacements = list()
	gridCall.nreplacements = list()

def a1111GridCallParamAddHook(gridCall, p, v):
	tempstring = cleanName(p)
	l1 = ['promptreplace','promptreplace1','promptreplace2','promptreplace3','promptreplace4','promptreplace5','promptreplace6','promptreplace7','promptreplace8','promptreplace9']
	if tempstring in l1:
		gridCall.replacements.append(v)
		return True
	return False

def a1111GridCallParamAddHookNeg(gridcall, p, v):
	tempstring = cleanName(p)
	l2 = ['negativepromptreplace','negativepromptreplace1','negativepromptreplace2','negativepromptreplace3','negativepromptreplace4','negativepromptreplace5','negativepromptreplace6','negativepromptreplace7','negativepromptreplace8','negativepromptreplace9', 'npromptreplace']
	if tempstring in l2:
		gridcall.nreplacements.append(v)
		return True
	return False

def a1111GridCallApplyHook(gridCall, p, dry):
	for replace in gridCall.replacements:
		applyPromptReplace(p, replace)
	for nreplace in gridCall.nreplacements:
		applyNegPromptReplace(p,nreplace)
	
def a1111GridRunnerPreRunHook(gridRunner):
	shared.total_tqdm.updateTotal(gridRunner.totalSteps)

class TempHolder: pass

def a1111GridRunnerPreDryHook(gridRunner):
	gridRunner.temp = TempHolder()
	gridRunner.temp.oldClipSkip = opts.CLIP_stop_at_last_layers
	gridRunner.temp.oldCodeformerWeight = opts.code_former_weight
	gridRunner.temp.oldFaceRestorer = opts.face_restoration_model
	gridRunner.temp.eta_noise_seed_delta = opts.eta_noise_seed_delta
	gridRunner.temp.oldVae = opts.sd_vae
	gridRunner.temp.oldModel = opts.sd_model_checkpoint

def a1111GridRunnerPostDryHook(gridRunner, promptkey: StableDiffusionProcessing, appliedsets: dict) -> Processed:
	promptkey.seed = processing.get_fixed_seed(promptkey.seed)
	promptkey.subseed = processing.get_fixed_seed(promptkey.subseed)
	processed = process_images(promptkey)
	#print(process_images)
	if len(processed.images) < 1:
		raise RuntimeError(f"Something went wrong! Image gen '{set.data}' produced {len(processed.images)} images, which is wrong")
	print(f"There are {len(processed.images)} images available in this set")
	for iterator, set in enumerate(appliedsets):
		os.makedirs(os.path.dirname(set.filepath), exist_ok=True)
	print(f"There are {len(appliedsets)} set applied")
	for iterator, img in enumerate(processed.images):
		print(f"currently saving image {iterator + 1} from current set")
		#img = processed.images[iterator]
		if iterator > len(list(appliedsets)) - 1: 
			print("image not in sets")
			continue
		set = list(appliedsets)[iterator]
		if len(promptkey.prompt) - 1 < iterator:
			print("image not in prompt list")
			continue
		if type(img) == numpy.ndarray:
			img = Image.fromarray(img)
		if hasattr(promptkey, 'inf_grid_out_width') and hasattr(promptkey, 'inf_grid_out_height'):
			img = img.resize((promptkey.inf_grid_out_width, promptkey.inf_grid_out_height), resample=images.LANCZOS)
		processed.images[iterator] = img
		info = processing.create_infotext(promptkey, [promptkey.prompt], [promptkey.seed], [promptkey.subseed], [])
		print(f"saving to: {os.path.dirname(set.filepath)}\\{os.path.basename(set.filepath)}")
		images.save_image(img, path=os.path.dirname(set.filepath), basename="", 
							forced_filename=os.path.basename(set.filepath), save_to_dirs=False, info=info, 
							extension=gridRunner.grid.format, p=promptkey, prompt=promptkey.prompt[iterator], seed=processed.seed)
	return processed

def a1111WebDataGetBaseParamData(p):
	return {
		"sampler": p.sampler_name,
		"seed": p.seed,
		"restorefaces": (opts.face_restoration_model if p.restore_faces else None),
		"steps": p.steps,
		"cfgscale": p.cfg_scale,
		"model": chooseBetterFileName('', shared.sd_model.sd_checkpoint_info.model_name).replace(',', '').replace(':', ''),
		"vae": (None if sd_vae.loaded_vae_file is None else (chooseBetterFileName('', sd_vae.loaded_vae_file).replace(',', '').replace(':', ''))),
		"width": p.width,
		"height": p.height,
		"prompt": p.prompt,
		"negativeprompt": p.negative_prompt,
		"varseed": (None if p.subseed_strength == 0 else p.subseed),
		"varstrength": (None if p.subseed_strength == 0 else p.subseed_strength),
		"clipskip": opts.CLIP_stop_at_last_layers,
		"codeformerweight": opts.code_former_weight,
		"denoising": getattr(p, 'denoising_strength', None),
		"eta": fixNum(p.eta),
		"sigmachurn": fixNum(p.s_churn),
		"sigmatmin": fixNum(p.s_tmin),
		"sigmatmax": fixNum(p.s_tmax),
		"sigmanoise": fixNum(p.s_noise),
		"ENSD": None if opts.eta_noise_seed_delta == 0 else opts.eta_noise_seed_delta
	}

class SettingsFixer():
	def __enter__(self):
		self.model = opts.sd_model_checkpoint
		self.CLIP_stop_at_last_layers = opts.CLIP_stop_at_last_layers
		self.code_former_weight = opts.code_former_weight
		self.face_restoration_model = opts.face_restoration_model
		self.eta_noise_seed_delta = opts.eta_noise_seed_delta
		self.vae = opts.sd_vae

	def __exit__(self, exc_type, exc_value, tb):
		opts.code_former_weight = self.code_former_weight
		opts.face_restoration_model = self.face_restoration_model
		opts.CLIP_stop_at_last_layers = self.CLIP_stop_at_last_layers
		opts.eta_noise_seed_delta = self.eta_noise_seed_delta
		opts.sd_vae = self.vae
		opts.sd_model_checkpoint = self.model
		sd_models.reload_model_weights()
		sd_vae.reload_vae_weights()

######################### Script class entrypoint #########################
class Script(scripts.Script):
	BASEDIR = scripts.basedir()
	VALIDATE_REPLACE = True

	def title(self):
		if __file__.endswith('.pyc'):
			return "Generate Infinite-Axis Grid (compiled)"
		else:
			return "Generate Infinite-Axis Grid (decompiled)"

	def show(self, is_img2img):
		return True

	def ui(self, is_img2img):
		core.listImageFiles()
		tryInit()
		gr.HTML(value=f"<br>Confused/new? View <a style=\"border-bottom: 1px #00ffff dotted;\" href=\"{INF_GRID_README}\">the README</a> for usage instructions.<br><br>")
		with gr.Row():
			grid_file = gr.Dropdown(value="Create in UI",label="Select grid definition file", choices=["Create in UI"] + core.getNameList())
			def refresh():
				newChoices = ["Create in UI"] + core.getNameList()
				grid_file.choices = newChoices
				return gr.update(choices=newChoices)
			refresh_button = ui_components.ToolButton(value=refresh_symbol, elem_id="infinity_grid_refresh_button")
			refresh_button.click(fn=refresh, inputs=[], outputs=[grid_file])
		output_file_path = gr.Textbox(value="", label="Output folder name (if blank uses yaml filename or current date)")
		page_will_be = gr.HTML(value="(...)<br><br>")
		manualGroup = gr.Group(visible=True)
		manualAxes = list()
		sets = list()
		with manualGroup:
			with gr.Row():
				with gr.Column():
					axisCount = 0
					for group in range(0, 4):
						groupObj = gr.Group(visible=group == 0)
						with groupObj:
							rows = list()
							for i in range(0, 4):
								with gr.Row():
									axisCount += 1
									row_mode = gr.Dropdown(value="", label=f"Axis {axisCount} Mode", choices=[""] + [x.name for x in core.validModes.values()])
									row_value = gr.Textbox(label=f"Axis {axisCount} Value", lines=1)
									fill_row_button = ui_components.ToolButton(value=fill_values_symbol, visible=False)
									def fillAxis(modeName):
										core.clearCaches()
										mode = core.validModes.get(cleanName(modeName))
										if mode is None:
											return gr.update()
										elif mode.type == "boolean":
											return "true, false"
										elif mode.valid_list is not None:
											return ", ".join(list(mode.valid_list()))
										raise RuntimeError(f"Can't fill axis for {modeName}")
									fill_row_button.click(fn=fillAxis, inputs=[row_mode], outputs=[row_value])
									def onAxisChange(modeName, outFile):
										mode = core.validModes.get(cleanName(modeName))
										buttonUpdate = gr.Button.update(visible=mode is not None and (mode.valid_list is not None or mode.type == "boolean"))
										outFileUpdate = gr.Textbox.update() if outFile != "" else gr.Textbox.update(value=f"autonamed_inf_grid_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}")
										return [buttonUpdate, outFileUpdate]
									row_mode.change(fn=onAxisChange, inputs=[row_mode, output_file_path], outputs=[fill_row_button, output_file_path])
									manualAxes += list([row_mode, row_value])
									rows.append(row_mode)
							sets.append([groupObj, rows])
		for group in range(0, 3):
			row_mode = sets[group][1][3]
			groupObj = sets[group + 1][0]
			nextRows = sets[group + 1][1]
			def makeVis(prior, r1, r2, r3, r4):
				return gr.Group.update(visible=prior+r1+r2+r3+r4 != "")
			row_mode.change(fn=makeVis, inputs=[row_mode] + nextRows, outputs=[groupObj])
		gr.HTML('<span style="opacity:0.5;">(More input rows will be automatically added after you select modes above.)</span>')
		grid_file.change(
			fn=lambda x: {"visible": x == "Create in UI", "__type__": "update"},
			inputs=[grid_file],
			outputs=[manualGroup],
			show_progress = False)
		def getPageUrlText(file):
			if file is None:
				return "(...)"
			outPath = opts.outdir_grids or (opts.outdir_img2img_grids if is_img2img else opts.outdir_txt2img_grids)
			fullOutPath = outPath + "/" + file
			return f"Page will be at <a style=\"border-bottom: 1px #00ffff dotted;\" href=\"/file={fullOutPath}/index.html\">(Click me) <code>{fullOutPath}</code></a><br><br>"
		def updatePageUrl(filePath, selectedFile):
			return gr.update(value=getPageUrlText(filePath or (selectedFile.replace(".yml", "") if selectedFile is not None else None)))
		output_file_path.change(fn=updatePageUrl, inputs=[output_file_path, grid_file], outputs=[page_will_be])
		grid_file.change(fn=updatePageUrl, inputs=[output_file_path, grid_file], outputs=[page_will_be])
		with gr.Row():
			do_overwrite = gr.Checkbox(value=False, label="Overwrite existing images (for updating grids)")
			dry_run = gr.Checkbox(value=False, label="Do a dry run to validate your grid file")
			#fast_skip = gr.Checkbox(value=True, label="Use more-performant skipping")
		with gr.Row():
			generate_page = gr.Checkbox(value=True, label="Generate infinite-grid webviewer page")
			validate_replace = gr.Checkbox(value=False, label="Validate PromptReplace input")
			publish_gen_metadata = gr.Checkbox(value=True, label="Publish full generation metadata for viewing on-page")
		return [do_overwrite, generate_page, dry_run, validate_replace, publish_gen_metadata, grid_file, output_file_path] + manualAxes

	def run(self, p, do_overwrite, generate_page, dry_run, validate_replace, publish_gen_metadata, grid_file, output_file_path, *manualAxes):
		core.clearCaches()
		tryInit()
		# Clean up default params
		p = copy(p)
		p.n_iter = 1
		#p.batch_size = 1
		p.do_not_save_samples = True
		p.do_not_save_grid = True
		p.seed = processing.get_fixed_seed(p.seed)
		# Store extra variable
		Script.VALIDATE_REPLACE = validate_replace
		# Validate to avoid abuse
		if '..' in grid_file or grid_file == "":
			raise RuntimeError(f"Unacceptable filename '{grid_file}'")
		if '..' in output_file_path:
			raise RuntimeError(f"Unacceptable alt file path '{output_file_path}'")
		if grid_file == "Create in UI":
			#if output_file_path is None or output_file_path == "":
			#	raise RuntimeError(f"Must specify the output file path")
			manualAxes = list(manualAxes)
		else:
			manualAxes = None
		with SettingsFixer():
			result = core.runGridGen(p, grid_file, p.outpath_grids, output_file_path, do_overwrite, generate_page, publish_gen_metadata, dry_run, manualAxes)
		if result is None:
			return Processed(p, list())
		return result
