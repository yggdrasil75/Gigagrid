import gradio as gr
import os, numpy, threading, ctypes, json, torch, hashlib, datetime
from copy import copy
from modules import images, shared, sd_models, sd_vae, sd_samplers, scripts, processing, ui_components
from modules.processing import StableDiffusionProcessing, process_images, Processed
from modules.shared import opts, state
from PIL import Image
import gigacore as core
from gigacore import getBestInList, cleanName, cleanModeName, chooseBetterFileName, GridSettingMode, fixNum, applyField, registerMode, applyOverride, applyGigaField
from colorama import init as CInit, Fore, Style
from quickcache import QuickCache

######################### Constants #########################
refresh_symbol = '\U0001f504'  # 🔄
fill_values_symbol = "\U0001f4d2"  # 📒
INF_GRID_README = "https://github.com/yggdrasil75/gigagrid"
core.EXTRA_FOOTER = 'Images area auto-generated by an AI (Stable Diffusion) and so may not have been reviewed by the page author before publishing.\n<script src="a1111webui.js?vary=9"></script>'
core.EXTRA_ASSETS = ["a1111webui.js"]
cplist = {}
faceRestorList = {}
vaeList = {}
adetailer = None

######################### Value Modes #########################

@QuickCache(100, "./cache/GMF")
def getModelFor(name):
	return getBestInList(name, cplist)

@QuickCache(100, "./cache/cleanmodel")
def cleanModel(p, v):
	actualModel = getBestInList(v, cplist)
	if actualModel is None:
		raise RuntimeError(f"Invalid parameter '{p}' as '{v}': model name unrecognized - valid {list(cplist)}")
	return chooseBetterFileName(v, actualModel)

def applyStyles(p,v:str):
	@QuickCache(100, "./cache/style")
	def getStyle(new, existing):
		if existing is not None:
			return existing.append(list(new.split(',')))
		else:
			return list(new.split(','))
	p.styles = getStyle(v, p.styles or None)

def cleanVae(p, v):
	vaeName = cleanName(v)
	if vaeName == "auto": return "automatic"
	if vaeName in ["none", "automatic"]:
		return vaeName
	actualVae = getBestInList(vaeName, vaeList)
	if actualVae is None:
		raise RuntimeError(f"Invalid parameter '{p}' as '{v}': VAE name unrecognized - valid: {vaeList}")
	return chooseBetterFileName(v, actualVae)

def applyCodeformerWeight(p, v):
	opts.code_former_weight = float(v)

def applyRestoreFaces(p, v):
	global faceRestorList
	input = cleanName(v)
	if input == "false":
		p.restore_faces = False
		return
	p.restore_faces = True
	restorer = getBestInList(input, faceRestorList)
	if restorer is not None:
		opts.face_restoration_model = restorer

def applyPromptReplace(p, v):
	if not '\n' in v:
		val = v.split('=', maxsplit=1)
		if len(val) != 2:
			raise RuntimeError(f"Invalid prompt replace, missing '=' symbol, for '{v}'")
		match = val[0].strip()
		replace = val[1].strip()
		if Script.VALIDATE_REPLACE:
			if match not in p.prompt:
				raise RuntimeError(f"Invalid prompt replace, '{match}' is not in prompt '{p.prompt}' nor negative prompt '{p.negative_prompt}'")
		p.prompt = p.prompt.replace(match, replace)
	else:
		valList = list(v)
		for valitem in v:
			val = v.split('=', maxsplit=1)
			if len(val) != 2:
				raise RuntimeError(f"Invalid prompt replace, missing '=' symbol, for '{v}'")
			match = val[0].strip()
			replace = val[1].strip()
			if Script.VALIDATE_REPLACE:
				if match not in p.prompt and match not in p.negative_prompt:
					raise RuntimeError(f"Invalid prompt replace, '{match}' is not in prompt '{p.prompt}' nor negative prompt '{p.negative_prompt}'")
			p.prompt = p.prompt.replace(match, replace)
			
def applyNegPromptReplace(p, v):
	if not '\n' in v:
		val = v.split('=', maxsplit=1)
		if len(val) != 2:
			raise RuntimeError(f"Invalid prompt replace, missing '=' symbol, for '{v}'")
		match = val[0].strip()
		replace = val[1].strip()
		if Script.VALIDATE_REPLACE:
			if match not in p.negative_prompt:
				raise RuntimeError(f"Invalid prompt replace, '{match}' is not in negative prompt '{p.negative_prompt}'")
		p.negative_prompt = p.negative_prompt.replace(match, replace)
	else:
		valList = list(v)
		for valitem in valList:
			val = valitem.split('=', maxsplit=1)
			if len(val) != 2:
				raise RuntimeError(f"Invalid prompt replace, missing '=' symbol, for '{valitem}'")
			match = val[0].strip()
			replace = val[1].strip()
			if Script.VALIDATE_REPLACE:
				if match not in p.negative_prompt:
					raise RuntimeError(f"Invalid prompt replace, '{match}' is not in negative prompt '{p.negative_prompt}'")
			p.negative_prompt = p.negative_prompt.replace(match, replace)

def applyEnableHr(p, v):
	p.enable_hr = v
	if v:
		if p.denoising_strength is None:
			p.denoising_strength = 0.75
		
def setbatch(p, v: str):
	# Load the model information from the JSON file
	# Get the absolute path of the directory in which the script is running
	#dir_path = os.path.abspath(os.path.dirname(__file__))
	#print(dir_path)
	# Load the JSON file from the current directory
	#with open(os.path.join(dir_path, 'models.json')) as f:
	#	models_data = json.load(f)

	# Get the GPU information
	#gpu = torch.cuda.get_device_name(torch.cuda.current_device)
	#print(gpu)
	#architecture = getarch()
	#print(architecture)
	#ram = torch.cuda.get_device_properties(torch.cuda.current_device).total_memory / 1024**3  # Convert to GB

	try:
		p.batch_size = int(v)
	except: core.DataLog("error in batch Size. check that it is a number", True, 0)

def adetailerField(p,v):
	return

def enableAdetailer(p,v):
	if v == True:
		p.hasPostProcessing = True
		p._ad_disabled = True
		postscript = adetailer.AfterDetailerScript()
		postscript.is_ad_enabled()
		return

def applyADetailer(p,v):
	p.is_ad_enabled
	return

######################### Addons #########################
hasInited = False

def tryInit():
	global cplist, faceRestorList, vaeList
	cplist = list(map(lambda m: m.title, sd_models.checkpoints_list.values()))
	faceRestorList = list(map(lambda m: m.name(), shared.face_restorers))
	vaeList = sd_vae.vae_dict.keys()
	core.getModelFor = getModelFor
	core.gridCallInitHook = a1111GridCallInitHook
	core.gridCallParamAddHook = a1111GridCallParamAddHook
	core.gridCallApplyHook = a1111GridCallApplyHook
	core.gridRunnerPreRunHook = a1111GridRunnerPreRunHook
	core.gridRunnerPreDryHook = a1111GridRunnerPreDryHook
	core.gridRunnerRunPostDryHook = a1111GridRunnerPostDryHook
	core.webDataGetBaseParamData = a1111WebDataGetBaseParamData
	registerMode("Model", GridSettingMode(dry=False, type=str, apply=applyOverride("sd_model_checkpoint"), clean=cleanModel, validList=lambda: list(map(lambda m: m.title, sd_models.checkpoints_list.values()))))
	registerMode("VAE", GridSettingMode(dry=False, type=str, apply=applyOverride("sd_vae"), clean=cleanVae, validList=lambda: list(sd_vae.vae_dict.keys()) + ['none', 'auto', 'automatic']))
	registerMode("Sampler", GridSettingMode(dry=True, type=str, apply=applyField("sampler_name"), validList=lambda: list(sd_samplers.all_samplers_map.keys())))
	registerMode("ClipSkip", GridSettingMode(dry=False, type=int, min=1, max=12, apply=applyOverride("CLIP_stop_at_last_layers")))
	registerMode("Restore Faces", GridSettingMode(dry=True, type=str, apply=applyRestoreFaces, validList=lambda: list(map(lambda m: m.name(), shared.face_restorers)) + ["true", "false"]))
	registerMode("CodeFormer Weight", GridSettingMode(dry=True, type=float, min=0, max=1, apply=applyCodeformerWeight))
	registerMode("ETA Noise Seed Delta", GridSettingMode(dry=True, type=int, apply=applyOverride("eta_noise_seed_delta")))
	registerMode("Enable HighRes Fix", GridSettingMode(dry=True, type=bool, apply=applyEnableHr))
	registerMode("Prompt Replace", GridSettingMode(dry=True, type=str, apply=applyPromptReplace))
	registerMode("Negative Prompt Replace", GridSettingMode(dry=True, type=str, apply=applyNegPromptReplace))
	registerMode("N Prompt Replace", GridSettingMode(dry=True, type=str, apply=applyNegPromptReplace))
	for i in range(0, 10):
		registerMode(f"Prompt Replace{i}", GridSettingMode(dry=True, type=str, apply=applyPromptReplace))
		registerMode(f"Negative Prompt Replace{i}", GridSettingMode(dry=True, type=str, apply=applyNegPromptReplace))
		registerMode(f"N Prompt Replace{i}", GridSettingMode(dry=True, type=str, apply=applyNegPromptReplace))
	modes = ["var seed","seed", "width", "height"]
	fields = ["subseed", "seed",  "width", "height"]
	for field, mode in enumerate(modes):
		registerMode(mode, GridSettingMode(dry=True, type=int, apply=applyField(fields[field])))
	modes = ["prompt", "negative prompt"]
	fields = ["prompt", "negative_prompt"]
	for field, mode in enumerate(modes):
		registerMode(mode, GridSettingMode(dry=True, type=str, apply=applyField(fields[field])))
	
	registerMode("Styles", GridSettingMode(dry=True, type=str, apply=applyStyles, validList=lambda: list(shared.prompt_styles.styles)))
	registerMode("batch size", GridSettingMode(dry=True, type=str, apply=setbatch))
	registerMode("Steps", GridSettingMode(dry=True, type=int, min=0, max=200, apply=applyField("steps")))
	registerMode("CFG Scale", GridSettingMode(dry=True, type=float, min=0, max=500, apply=applyField("cfg_scale")))
	registerMode("Tiling", GridSettingMode(dry=True, type=bool, apply=applyField("tiling")))
	registerMode("Var Strength", GridSettingMode(dry=True, type=float, min=0, max=1, apply=applyField("subseed_strength")))
	registerMode("Denoising", GridSettingMode(dry=True, type=float, min=0, max=1, apply=applyField("denoising_strength")))
	registerMode("ETA", GridSettingMode(dry=True, type=float, min=0, max=1, apply=applyField("eta")))
	registerMode("Sigma Churn", GridSettingMode(dry=True, type=float, min=0, max=1, apply=applyField("s_churn")))
	registerMode("Sigma TMin", GridSettingMode(dry=True, type=float, min=0, max=1, apply=applyField("s_tmin")))
	registerMode("Sigma TMax", GridSettingMode(dry=True, type=float, min=0, max=1, apply=applyField("s_tmax")))
	registerMode("Sigma Noise", GridSettingMode(dry=True, type=float, min=0, max=1, apply=applyField("s_noise")))
	registerMode("Out Width", GridSettingMode(dry=True, type=int, min=0, apply=applyGigaField("inf_grid_out_width")))
	registerMode("Out Height", GridSettingMode(dry=True, type=int, min=0, apply=applyGigaField("inf_grid_out_height")))
	registerMode("Out Scale", GridSettingMode(dry=True, type=float, min=0, apply=applyGigaField("inf_grid_out_scale")))
	registerMode("group", GridSettingMode(dry=True, type=int, min=0, apply=applyGigaField("sortgroup")))

	#not finalized. this sets the number of other items that will be compared to each to merge based on steps.
	#This will be set to 1000 by default to prevent n^2 at high numbers being slow and instead become 1000^2*n (or something like that)
	registerMode("Max Compare", GridSettingMode(dry=True, type=float, min=0, apply=applyGigaField("maxStepComp"))) 
	registerMode("Image Mask Weight", GridSettingMode(dry=True, type=float, min=0, max=1, apply=applyField("inpainting_mask_weight")))
	registerMode("HighRes Scale", GridSettingMode(dry=True, type=float, min=1, max=16, apply=applyField("hr_scale")))
	registerMode("HighRes Steps", GridSettingMode(dry=True, type=int, min=0, max=200, apply=applyField("hr_second_pass_steps")))
	registerMode("HighRes Resize Width", GridSettingMode(dry=True, type=int, apply=applyField("hr_resize_x")))
	registerMode("HighRes Resize Height", GridSettingMode(dry=True, type=int, apply=applyField("hr_resize_y")))
	registerMode("HighRes Upscale to Width", GridSettingMode(dry=True, type=int, apply=applyField("hr_upscale_to_x")))
	registerMode("HighRes Upscale to Height", GridSettingMode(dry=True, type=int, apply=applyField("hr_upscale_to_y")))
	registerMode("HighRes Upscaler", GridSettingMode(dry=True, type=str, apply=applyField("hr_upscaler"), validList=lambda: list(map(lambda u: u.name, shared.sd_upscalers)) + list(shared.latent_upscale_modes.keys())))
	registerMode("Image CFG Scale", GridSettingMode(dry=True, type=float, min=0, max=500, apply=applyField("image_cfg_scale")))
	registerMode("Use Result Index", GridSettingMode(dry=True, type=int, min=0, max=500, apply=applyGigaField("inf_grid_use_result_index")))

	try:
		scriptList = [x for x in scripts.scripts_data if x.script_class.__module__ == "dynamic_thresholding.py"][:1]
		if len(scriptList) == 1:
			dynamic_thresholding = scriptList[0].module
			registerMode("[DynamicThreshold] Enable", GridSettingMode(dry=True, type=bool, apply=applyField("dynthres_enabled")))
			registerMode("[DynamicThreshold] Mimic Scale", GridSettingMode(dry=True, type=float, min=0, max=500, apply=applyField("dynthres_mimic_scale")))
			registerMode("[DynamicThreshold] Threshold Percentile", GridSettingMode(dry=True, type=float, min=0.0, max=100.0, apply=applyField("dynthres_threshold_percentile")))
			registerMode("[DynamicThreshold] Mimic Mode", GridSettingMode(dry=True, type=str, apply=applyField("dynthres_mimic_mode"), validList=lambda: list(dynamic_thresholding.VALID_MODES)))
			registerMode("[DynamicThreshold] CFG Mode", GridSettingMode(dry=True, type=str, apply=applyField("dynthres_cfg_mode"), validList=lambda: list(dynamic_thresholding.VALID_MODES)))
			registerMode("[DynamicThreshold] Mimic Scale Minimum", GridSettingMode(dry=True, type=float, min=0.0, max=100.0, apply=applyField("dynthres_mimic_scale_min")))
			registerMode("[DynamicThreshold] CFG Scale Minimum", GridSettingMode(dry=True, type=float, min=0.0, max=100.0, apply=applyField("dynthres_cfg_scale_min")))
			registerMode("[DynamicThreshold] Experiment Mode", GridSettingMode(dry=True, type=float, min=0, max=1000000, apply=applyField("dynthres_experiment_mode")))
			registerMode("[DynamicThreshold] scheduler Value", GridSettingMode(dry=True, type=float, min=0, max=100, apply=applyField("dynthres_scheduler_val")))
			registerMode("[DynamicThreshold] Scaling Startpoint", GridSettingMode(dry=True, type=str, apply=applyField("dynthres_scaling_startpoint"), validList=lambda: list(['ZERO', 'MEAN'])))
			registerMode("[DynamicThreshold] Variability Measure", GridSettingMode(dry=True, type=str, apply=applyField("dynthres_variability_measure"), validList=lambda: list(['STD', 'AD'])))
			registerMode("[DynamicThreshold] Interpolate Phi", GridSettingMode(dry=True, type=float, min=0, max=1, apply=applyField("dynthres_interpolate_phi")))
			registerMode("[DynamicThreshold] Separate Feature Channels", GridSettingMode(dry=True, type=bool, apply=applyField("dynthres_separate_feature_channels")))
		scriptList = [x for x in scripts.scripts_data if x.script_class.__module__ == "controlnet.py"][:1]
		if len(scriptList) == 1:
			# Hacky and doesnt work. only works with the main controlnet and not any other implementations.
			try: 
				preprocessors_list = scriptList[0].script_class().preprocessor.keys() # if you see an error about this with sd forge, then just know that I dont have a fix right now.
			except:
				preprocessors_list = list()
			module = scriptList[0].module
			def validateParam(p, v):
				if not shared.opts.data.get("control_net_allow_script_control", False):
					raise RuntimeError("ControlNet options cannot currently work, you must enable 'Allow other script to control this extension' in Settings -> ControlNet first")
				return v
			registerMode("[ControlNet] Enable", GridSettingMode(dry=True, type=bool, apply=applyField("control_net_enabled"), clean=validateParam))
			registerMode("[ControlNet] Preprocessor", GridSettingMode(dry=True, type=str, apply=applyField("control_net_module"), clean=validateParam, validList=lambda: list(scriptList[0].script_class().preprocessor.keys())))
			registerMode("[ControlNet] Model", GridSettingMode(dry=True, type=str, apply=applyField("control_net_model"), clean=validateParam, validList=lambda: list(list(module.cn_models.keys()))))
			registerMode("[ControlNet] Weight", GridSettingMode(dry=True, type=float, min=0.0, max=2.0, apply=applyField("control_net_weight"), clean=validateParam))
			registerMode("[ControlNet] Guidance Strength", GridSettingMode(dry=True, type=float, min=0.0, max=1.0, apply=applyField("control_net_guidance_strength"), clean=validateParam))
			registerMode("[ControlNet] Annotator Resolution", GridSettingMode(dry=True, type=int, min=0, max=2048, apply=applyField("control_net_pres"), clean=validateParam))
			registerMode("[ControlNet] Threshold A", GridSettingMode(dry=True, type=int, min=0, max=256, apply=applyField("control_net_pthr_a"), clean=validateParam))
			registerMode("[ControlNet] Threshold B", GridSettingMode(dry=True, type=int, min=0, max=256, apply=applyField("control_net_pthr_b"), clean=validateParam))
			registerMode("[ControlNet] Image", GridSettingMode(dry=True, type=str, apply=core.applyFieldAsImageData("control_net_input_image"), clean=validateParam, validList=lambda: core.listImageFiles()))
		#scriptList = [x for x in scripts.scripts_data if x.script_class.__module__ == "!adetailer.py"][:1]
		#if len(scriptList) == 1:
		#	global adetailer
		#	adetailer = scriptList[0].module
		#	registerMode("[adetailer] enable", GridSettingMode(dry=True, type=bool, apply=applyADetailer))
		#	for i in [1,2]:
		#		registerMode("[adetailer] positive prompt {i}", GridSettingMode(dry=True, type=str, apply=adetailerField))
		#		registerMode("[adetailer] negative prompt {i}")
		#		for i2 in range(0,10):
		#			registerMode("[adetailer] prompt replace {i}.{i2}")
		#			registerMode("[adetailer] negative prompt replace {i}.{i2}")
		#		registerMode("[adetailer] model {i}")
		#		registerMode("[adetailer] confidence {i}")
		#		registerMode("[adetailer] top k {i}")
		#		registerMode("[adetailer] min ratio {i}")
		#		registerMode("[adetailer] max ratio {i}")
		#		registerMode("[adetailer] mask x offset {i}")
		#		registerMode("[adetailer] mask y offset {i}")
		#		registerMode("[adetailer] erosion {i}")
		#		registerMode("[adetailer] mask merge mode {i}")
		#		registerMode("[adetailer] inpaint blur {i}")
		#		registerMode("[adetailer] inpaint denoise {i}")
		#		registerMode("[adetailer] inpaint only masked {i}")
		#		registerMode("[adetailer] inpaint padding {i}")
		#		registerMode("[adetailer] inpaint width {i}")
		#		registerMode("[adetailer] inpaint height {i}")
		#		registerMode("[adetailer] steps {i}")
		#		registerMode("[adetailer] cfg {i}")
		#		registerMode("[adetailer] checkpoint {i}")
		#		registerMode("[adetailer] sampler {i}")
		#		registerMode("[adetailer] noise {i}")
		#		registerMode("[adetailer] clip {i}")
		#		registerMode("[adetailer] restore faces {i}")
    
    
	except ModuleNotFoundError as e:
		print(f"Infinity Grid Generator failed to import a dependency module: {e}")
		pass

######################### Actual Execution Logic #########################

def a1111GridCallInitHook(gridCall: core.SingleGridCall):
	gridCall.replacements = list()
	gridCall.nreplacements = list()

def a1111GridCallParamAddHook(gridCall: core.SingleGridCall,grid, p: str, v: str) -> bool:
	if grid.minWidth is None:
		grid.minWidth = grid.initialPromptskey.width
	if grid.minHeight is None:
		grid.minHeight = grid.initialPromptskey.height
	tempstring = cleanModeName(p)
	if tempstring.startswith('promptreplace'):
		gridCall.replacements.append(v)
		return True
	elif tempstring.startswith('npromptreplace') or tempstring.startswith('negativepromptreplace'):
		gridCall.nreplacements.append(v)
		return True
	elif tempstring in ['width', 'outwidth']:
		grid.minWidth = min(grid.minWidth, v)
		return True
	elif tempstring in ['height', 'outheight']:
		grid.minHeight = min(grid.minHeight, v)
		return True
	return False

def a1111GridCallApplyHook(gridCall: core.SingleGridCall, p: str):
	for replace in gridCall.replacements:
		applyPromptReplace(p, replace)
	for nreplace in gridCall.nreplacements:
		applyNegPromptReplace(p,nreplace)
	
def a1111GridRunnerPreRunHook(gridRunner: core.GridRunner):
	state.job_count = gridRunner.totalRun
	shared.total_tqdm.updateTotal(gridRunner.totalSteps)
	state.processing_has_refined_job_count = True

class TempHolder: pass

def a1111GridRunnerPreDryHook(gridRunner: core.GridRunner):
	gridRunner.temp = TempHolder()
	gridRunner.temp.oldCodeformerWeight = opts.code_former_weight
	gridRunner.temp.oldFaceRestorer = opts.face_restoration_model
	gridRunner.temp.oldVae = opts.sd_vae
	gridRunner.temp.oldModel = opts.sd_model_checkpoint

def a1111GridRunnerPostDryHook(gridRunner: core.GridRunner, promptkey: StableDiffusionProcessing) -> Processed:
	processed: Processed = process_images(promptkey)
	#print(process_images)
	if len(processed.images) < 1:
		raise RuntimeError(f"Something went wrong! Image gen '{set.data}' produced {len(processed.images)} images, which is wrong")
	print(f"There are {len(processed.images)} images available in this set")
	for iterator, img in enumerate(processed.images):
		if len(promptkey.prompt) - 1 < iterator:
			print("image not in prompt list")
			continue
		if type(img) == numpy.ndarray:
			img = Image.fromarray(img)
		if 'inf_grid_out_width' in promptkey.giga or 'inf_grid_out_height' in promptkey.giga:
			original_width, original_height = img.size
			scale_factor = promptkey.giga.get('inf_grid_out_scale', 1.0)
			
			new_width = promptkey.giga.get('inf_grid_out_width', original_width) * scale_factor
			new_height = promptkey.giga.get('inf_grid_out_height', original_height) * scale_factor

			if 'inf_grid_out_width' in promptkey.giga and 'inf_grid_out_height' not in promptkey.giga:
				new_height = int(original_height * (new_width / original_width))
			elif 'inf_grid_out_height' in promptkey.giga and 'inf_grid_out_width' not in promptkey.giga:
				new_width = int(original_width * (new_height / original_height))

			img = img.resize((int(new_width), int(new_height)), resample=images.LANCZOS)

		processed.images[iterator] = img
		#images.save_image(img, path=os.path.dirname(set.filepath), basename="",
		#	forced_filename=os.path.basename(set.filepath), save_to_dirs=False,info=info,
		#	extension=gridRunner.grid.format, p=promptkey, prompt=promptkey.prompt[iterator],seed=processed.seed)
		#threading.Thread(target=saveOffThread).start()
	return processed

def a1111WebDataGetBaseParamData(p) -> dict:
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
		self.code_former_weight = opts.code_former_weight
		self.face_restoration_model = opts.face_restoration_model
		self.vae = opts.sd_vae

	def __exit__(self, exc_type, exc_value, tb):
		opts.code_former_weight = self.code_former_weight
		opts.face_restoration_model = self.face_restoration_model
		opts.sd_vae = self.vae
		opts.sd_model_checkpoint = self.model
		sd_models.reload_model_weights()
		sd_vae.reload_vae_weights()

######################### Script class entrypoint #########################
class Script(scripts.Script):
	BASEDIR = scripts.basedir()
	VALIDATE_REPLACE = True

	def title(self) -> str:
		if __file__.endswith('.pyc'):
			return "Generate Infinite-Axis Grid (compiled)"
		else:
			return "Generate Infinite-Axis Grid v2"

	def show(self, is_img2img) -> bool:
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
		def getPageUrlText(file):
			if file is None:
				return "(...)"
			out_path = opts.outdir_grids or (opts.outdir_img2img_grids if is_img2img else opts.outdir_txt2img_grids)
			full_out_path = out_path + "/" + file
			notice = ""
			if os.path.exists(full_out_path):
				notice = "<br><span style=\"color: red;\">NOTICE: There is already something saved there! This will overwrite prior data.</span>"
			return f"Page will be at <a style=\"border-bottom: 1px #00ffff dotted;\" href=\"/file={full_out_path}/index.html\">(Click me) <code>{full_out_path}</code></a>{notice}<br><br>"
		def update_page_url(file_path, selected_file):
			return gr.update(value=getPageUrlText(file_path or (selected_file.replace(".yml", "") if selected_file is not None else None)))
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
									row_mode = gr.Dropdown(value="", label=f"Axis {axisCount} Mode", choices=[" "] + [x.name for x in core.validModes.values()])
									row_value = gr.Textbox(label=f"Axis {axisCount} Value", lines=1)
									fill_row_button = ui_components.ToolButton(value=fill_values_symbol, visible=False)
									def fillAxis(modeName):
										mode = core.validModes.get(cleanModeName(modeName))
										if mode is None:
											return gr.update()
										elif mode.type == "boolean":
											return "true, false"
										elif mode.valid_list is not None:
											return ", ".join(list(mode.valid_list()))
										raise RuntimeError(f"Can't fill axis for {modeName}")
									fill_row_button.click(fn=fillAxis, inputs=[row_mode], outputs=[row_value])
									def onAxisChange(modeName, outFile):
										mode = core.validModes.get(cleanModeName(modeName))
										buttonUpdate = gr.Button.update(visible=mode is not None and (mode.valid_list is not None or mode.type == "boolean"))
										outFileUpdate = gr.Textbox.update() if outFile != "" else gr.Textbox.update(value=f"autonamed_inf_grid_{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}")
										notice = update_page_url(outFile, None)
										return [buttonUpdate, outFileUpdate, notice]
									row_mode.change(fn=onAxisChange, inputs=[row_mode, output_file_path], outputs=[fill_row_button, output_file_path, page_will_be])
									manualAxes += list([row_mode, row_value])
									rows.append(row_mode)
							sets.append([groupObj, rows])
		for group in range(0, 3):
			row_mode = sets[group][1][3]
			groupObj = sets[group + 1][0]
			nextRows = sets[group + 1][1]
			def makeVis(prior, r1, r2, r3, r4):
				return gr.Group.update(visible=(prior+r1+r2+r3+r4).strip() != "")
			row_mode.change(fn=makeVis, inputs=[row_mode] + nextRows, outputs=[groupObj])
		gr.HTML('<span style="opacity:0.5;">(More input rows will be automatically added after you select modes above.)</span>')
		grid_file.change(
			fn=lambda x: {"visible": x == "Create in UI", "__type__": "update"},
			inputs=[grid_file],
			outputs=[manualGroup],
			show_progress = False)
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
		starttime = datetime.datetime.now()
		print(f"The full process has started at {starttime}")
		tryInit()
		# Clean up default params
		p = copy(p)
		p.n_iter = 1
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
		#core.logFile = os.path.join(output_file_path, 'log.txt')
		p1time = datetime.datetime.now()
		print(f"finished everything before running in {(p1time - starttime).total_seconds():.2f}")
		with SettingsFixer():
			result = core.runGridGen(p, grid_file, p.outpath_grids, output_file_path, do_overwrite, generate_page, publish_gen_metadata, dry_run, manualAxes)
		if result is None:
			return Processed(p, list())
		return result
