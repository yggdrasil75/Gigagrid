# This file is part of Infinity Grid Generator, view the README.md at https://github.com/mcmonkeyprojects/sd-infinity-grid-generator-script for more information.

import os, glob, yaml, json, shutil, math, re, threading, hashlib, types, datetime, re, atexit, signal, multiprocessing, html as HTMLModule, logging, sys
from multiprocessing import Pool, cpu_count as cpuCount
from modules import sd_models as sdModels, images, processing
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img, Processed
from modules.shared import opts
from copy import copy, deepcopy
from PIL import Image
from git import Repo
from colorama import init as CInit, Fore, Style
from quickcache import QuickCache
CInit()

######################### Core Variables #########################

AssetDir: str = os.path.join(os.path.dirname(__file__), "assets")
ExtraFooter = "..."
ExtraAssets: list = []
validModes: dict = {}
ImagesCache = None
modelchange: dict = {}
logFile: str
assetFile:str = None
cacheFile: str = None
Version:str = '24.2.26'
printLevel: int= 1
grid = None
quickListCache:dict = {}
lock = threading.Lock()
cleanList: dict = {}
MixModels: list = []
logger = None

######################### Hooks #########################

# hook(getmodelfor)
getModelFor: callable = None
# hook(SingleGridCall)
gridCallInitHook: callable = None
# hook(SingleGridCall, paramName: str, value: any) -> bool
gridCallParamAddHook: callable = None
# hook(SingleGridCall, paramName: str, dry: bool)
gridCallApplyHook: callable = None
# hook(GridRunner)
gridRunnerPreRunHook: callable = None
# hook(GridRunner)
gridRunnerPreDryHook: callable = None
# hook(GridRunner, PassThroughObject, set: list(SingleGridCall)) -> ResultObject
gridRunnerRunPostDryHook: callable = None
# hook(PassThroughObject) -> dict
webDataGetBaseParamData: callable = None

######################### Utilities #########################

def DataLog(message: str, doprint: bool, level: int) -> None:
	try:
		if not os.path.exists(logFile):
			open(logFile, 'x')
		with open(logFile, 'a+', encoding="utf-8") as f:
			f.write(str(message))
			f.write('\n')
	except: print(f"[datalog] used before defining logFile, value: {str(message)}")
	
	if doprint: # and printLevel >= level:
		if level == 0: #warning
			print(Fore.RED + str(message) + Style.RESET_ALL, file=sys.stderr)
		elif level == 1: #info
			print(Fore.GREEN + str(message) + Style.RESET_ALL)
		elif level == 2: #debug
			print(Fore.BLUE + str(message) + Style.RESET_ALL)
	try:
		return message
	except:
		pass
	
def DataLogSetup(folder) -> str:
	logFile = os.path.join(folder, 'log.txt')

	# Create a list of existing log files in the folder
	logFiles = [logFile] + [os.path.join(folder, f'log{i}.txt') for i in range(1, 5)]

	# Rename existing log files in reverse order
	for i in range(5, 1, -1):
		src = os.path.join(folder, f'log{i - 1}.txt')
		dest = os.path.join(folder, f'log{i}.txt')
		if dest.endswith("log5.txt") and os.path.exists(dest):
			os.remove(dest)
		if os.path.exists(src):
			os.rename(src, dest)

	# Rename the current log file to 'log1.txt'
	if os.path.exists(logFile):
		os.rename(logFile, os.path.join(folder, 'log1.txt'))

	return logFile

def quickTimeMath(startTime:datetime.datetime, endTime:datetime.datetime = None) -> str:
	if endTime == None: endTime = datetime.datetime.now()
	return "{:.2f}".format((endTime - startTime).total_seconds())

def getVersion() -> str:
	global Version
	if Version is not None:
		return Version
	try:
		repo = Repo(os.path.dirname(__file__))
		Version = repo.head.commit.hexsha[:8]
	except Exception:
		Version = "Unknown"
	return Version

def listImageFiles() -> list:
	global ImagesCache
	starttime = datetime.datetime.now()
	if ImagesCache is not None:
		return ImagesCache
	imageDir = os.path.join(AssetDir, "images")
	ImagesCache = list()
	for path, _, files in os.walk(imageDir):
		for name in files:
			_, ext = os.path.splitext(name)
			if ext in [".jpg", ".png", ".webp"]:
				fn = os.path.relpath(os.path.normcase(os.path.normpath(os.path.join(path, name))), imageDir)
				while fn.startswith('/'):
					fn = fn[1:]
				ImagesCache.append(fn)
	DataLog(f"[listImageFiles] done in {quickTimeMath(startTime=starttime)}", False, 2)
	return ImagesCache

def getNameList() -> list[str]:
    fileList = glob.glob(os.path.join(AssetDir, "*.yml"))
    fileList.extend(glob.glob(os.path.join(opts.outdir_txt2img_grids, "*.yml")))
    fileList.extend(glob.glob(os.path.join(opts.outdir_grids, "*.yml")))
    fileList.extend(glob.glob(os.path.join(opts.outdir_img2img_grids, "*.yml")))
    justFileNames = sorted(list(map(lambda f: os.path.relpath(f, AssetDir), fileList)))
    return justFileNames

@QuickCache(1000,"./cache/common")
def fixDict(d: dict):
	if d is None:
		return None
	if type(d) is not dict:
		raise RuntimeError(f"Value '{d}' is supposed to be submapping but isn't (it's plaintext, a list, or some other incorrect format). Did you typo the formatting?")
	return {str(k).lower(): v for k, v in d.items()}

#@QuickCache(1000,"./cache/common")
def cleanForWeb(text: str):
	if text is None:
		return None
	if type(text) is not str:
		raise RuntimeError(f"Value '{text}' is supposed to be text but isn't (it's a datamapping, list, or some other incorrect format). Did you typo the formatting?")
	return text.replace('"', '&quot;')

@QuickCache(1000,"./cache/common")
def cleanId(id: str) -> str:
	return re.sub("[^a-z0-9]", "_", id.lower().strip())

@QuickCache(1000, "./cache/common")
def cleanModeName(name: str) -> str:
    return name.lower().replace('[', '').replace(']', '').replace(' ', '').replace('\\', '/').replace('//', '/').strip()
	
@QuickCache(1000, "./cache/common")
def cleanName(name: str) -> str:
	cleanedName = re.sub(r'\[.*?\]', '', name)
	cleanedName = cleanedName.lower().replace(' ', '').replace('\\', '/').replace('//', '/').strip()
	return cleanedName

#@QuickCache(maxSize=1000,localCache="./cache/quickList")
def getQuickList(list:frozenset):
	global quickListCache
    # Calculate a hash for the input list.
	listHash = hashlib.blake2s(str(list).encode()).hexdigest()

    # Check if the list is already cached, and if so, return it.
	if listHash in quickListCache:
		DataLog(f"[getQuickList] using cache", doprint=False, level=2)
		return quickListCache[listHash]
	else: 
		DataLog(f"[getQuickList] not using cache", doprint=False, level=2)

		# If not cached, calculate the quick list.
		quickList = {}
		for item in list:
			normalizedItem = cleanName(item)
			if normalizedItem not in quickList:
				quickList[normalizedItem] = [item]
			else:
				quickList[normalizedItem].append(item)

		# Cache the quick list.
		quickListCache[listHash] = quickList

		return quickList

#@QuickCache(maxSize=1000,localCache="./cache/quickList")
def getBestInList(name, list):
	clean = cleanName(name)
	#dataLog(f"[getBestInList] name = {name} clean = {clean}", doprint=False, level=2)
    # Get the quick list for the input list.
	quickList = getQuickList(frozenset(list))
	if clean in quickList:
		return quickList[clean][0]
	else:
		return None

@QuickCache(maxSize=1000, localCache="./cache/betterName")
def chooseBetterFileName(rawName: str, fullName: str) -> str:
	#starttime = datetime.datetime.now()
	partialName = os.path.splitext(os.path.basename(fullName))[0]
	if '/' in rawName or '\\' in rawName or '.' in rawName or len(rawName) >= len(partialName):
		#DataLog(f"[chooseBetterFileName] done in {quickTimeMath(startTime=starttime)}", False, 2)
		return rawName
	#DataLog(f"[chooseBetterFileName] done in {quickTimeMath(startTime=starttime)}", False, 2)
	return partialName

@QuickCache(maxSize=1000, localCache="./cache/fixNum")
def fixNum(num) -> float | None:
	if num is None or math.isinf(num) or math.isnan(num):
		return None
	return num

@QuickCache(maxSize=1000, localCache="./cache/Numeric")
def expandNumericListRanges(inList, numType) -> list:
	outList = list()
	for i in range(0, len(inList)):
		rawVal = str(inList[i]).strip()
		if rawVal in ["..", "...", "....", "…"]:
			if i < 2 or i + 1 >= len(inList):
				raise RuntimeError(f"Cannot use ellipses notation at index {i}/{len(inList)} - must have at least 2 values before and 1 after.")
			prior = outList[-1]
			doublePrior = outList[-2]
			after = numType(inList[i + 1])
			step = prior - doublePrior
			if (step < 0) != ((after - prior) < 0):
				raise RuntimeError(f"Ellipses notation failed for step {step} between {prior} and {after} - steps backwards.")
			count = int((after - prior) / step)
			for x in range(1, count):
				outList.append(prior + x * step)
		else:
			outList.append(numType(rawVal))
	return outList

######################### Value Modes #########################

class GridSettingMode:
	"""
	Defines a custom parameter input mode for an Infinity Grid Generator.
	'dry' is True if the mode should be processed in dry runs, or False if it should be skipped.
	'type' is 'text', 'integer', 'decimal', or 'boolean'
	'apply' is a function to call taking (passthroughObject, value)
	'min' is for integer/decimal type, optional minimum value
	'max' is for integer/decimal type, optional maximum value
	'clean' is an optional function to call that takes (passthroughObject, value) and returns a cleaned copy of the value, or raises an error if invalid
	'validList' is for text type, an optional lambda that returns a list of valid values
	"""
	def __init__(self, dry: bool, type: type, apply: callable, min: float = None, max: float = None, validList: callable = None, clean: callable = None):
		self.dry = dry
		self.type = type
		self.apply = apply
		self.min = min
		self.max = max
		self.clean = clean
		self.validList = validList

def registerMode(name: str, mode: GridSettingMode) -> None:
	mode.name = name
	validModes[cleanModeName(name)] = mode

######################### Validation #########################

#@quickCache(maxsize=1000, localcache="./cache/validParams")
def validateParams(grid, params: dict) -> None:
	for p,v in params.items():
		params[p] = validateSingleParam(p, grid.procVariables(v))

#@quickCache(maxsize=1000, localcache="./cache/validateParams")
def validateSingleParam(p: str, value):
	p = cleanModeName(p)
	valuePrint = value
	mode = validModes.get(p)
	if mode is None:
		raise RuntimeError(f"Invalid grid parameter '{p}': unknown mode")
	modeType = mode.type
	if modeType == int:
		vInt = int(value)
		if vInt is None:
			raise RuntimeError(f"Invalid parameter '{p}' as '{valuePrint}': must be an integer number")
		min = mode.min
		max = mode.max
		if min is not None and vInt < min:
			raise RuntimeError(f"Invalid parameter '{p}' as '{valuePrint}': must be at least {min}")
		if max is not None and vInt > max:
			raise RuntimeError(f"Invalid parameter '{p}' as '{valuePrint}': must not exceed {max}")
		value = vInt
	elif modeType == float:
		vFloat = float(value)
		if vFloat is None:
			raise RuntimeError(f"Invalid parameter '{p}' as '{valuePrint}': must be a decimal number")
		min = mode.min
		max = mode.max
		if min is not None and vFloat < min:
			raise RuntimeError(f"Invalid parameter '{p}' as '{valuePrint}': must be at least {min}")
		if max is not None and vFloat > max:
			raise RuntimeError(f"Invalid parameter '{p}' as '{valuePrint}': must not exceed {max}")
		value = vFloat
	elif modeType == bool:
		vClean = str(value).lower().strip()
		if vClean == "true":
			retvalue = True
		elif vClean == "false":
			retvalue = False
		else:
			raise RuntimeError(f"Invalid parameter '{p}' as '{valuePrint}': must be either 'true' or 'false'")
	elif modeType == str and mode.validList is not None:
		validList = mode.validList()
		value = getBestInList(value, validList)
		if value is None:
			raise RuntimeError(f"Invalid parameter '{p}' as '{valuePrint}': not matched to any entry in list {list(validList)}")
	if mode.clean is not None:
		return mode.clean(p, value)
	return value

def applyField(name: str):
	def applier(p, v):
		setattr(p, name, v)
	return applier

def applyOverride(name: str):
	def applier(p, v):
		if not p.override_settings: setattr(p, 'override_settings', {})
		p.override_settings[name] = v
	return applier

def applyGigaField(name: str):
	def applier(p, v):
		if not hasattr(p, 'giga'):
			setattr(p, 'giga', {})
		p.giga[name] = v
	return applier

def applyFieldAsImageData(name: str):
	def applier(p, v):
		fName = getBestInList(v, listImageFiles())
		if fName is None:
			raise RuntimeError("Invalid parameter '{p}' as '{v}': image file does not exist")
		path = os.path.join(AssetDir, 'images', fName)
		image = Image.open(path)
		setattr(p, name, image)
	return applier


######################### YAML Parsing and Processing #########################

class AxisValue:
	def __init__(self, axis, grid, key: str, val) -> None:
		self.axis: Axis = axis
		self.key: str = cleanId(str(key))
		self.path = self.key
		if any(x.key == self.key for x in axis.values):
			self.key += f"__{len(axis.values)}"
		self.params: dict = {}
		if isinstance(val, str):
			halves = val.split('=', maxsplit=1)
			if len(halves) != 2:
				raise RuntimeError(f"Invalid value '{key}': '{val}': not expected format")
			halves[0] = grid.procVariables(halves[0])
			halves[1] = grid.procVariables(halves[1])
			halves[1] = validateSingleParam(halves[0], halves[1])
			self.title = halves[1]
			self.params = { cleanName(halves[0]): halves[1] }
			self.description = None
			self.skip = False
			self.show = True
		else:
			self.title: str = grid.procVariables(val.get("title"))
			self.description = grid.procVariables(val.get("description"))
			self.skip = False
			self.skipList: bool | dict = val.get("skip")
			if val.get("path") is not None:
				self.path = str(val.get("path"))
			else:
				self.path = cleanName(self.key)
			if isinstance(self.skipList, bool):
				self.skip = self.skipList
			elif self.skipList is not None and isinstance(self.skipList, dict):
				self.skip = self.skipList.get("always")
			self.params = fixDict(val.get("params"))
			self.show = (str(grid.procVariables(val.get("show")))).lower() != "false"
			if self.title is None or self.params is None:
				raise RuntimeError(f"Invalid value '{key}': '{val}': missing title or params")
			if not self.skip:
				threading.Thread(target=validateParams(grid, self.params)).start()
	
	def __str__(self) -> str:
		return f"(title={self.title}, description={self.description}, params={self.params})"
	def __unicode__(self) -> str:
		return self.__str__()
	
	def getPath(self) -> str:
		if self.path and self.path is not None:
			return self.path
		else:
			return cleanName(self.key)

class Axis:
	def __init__(self, grid, id: str, obj):
		self.rawID = id
		self.values = list()
		self.id = cleanId(str(id))
		if any(x.id == self.id for x in grid.axes):
			self.id += f"__{len(grid.axes)}"
		if isinstance(obj, str):
			self.title = id
			self.default = None
			self.description = ""
			self.buildFromListStr(id, grid, obj)
		else:
			self.title = grid.procVariables(obj.get("title"))
			self.default = grid.procVariables(obj.get("default"))
			if self.title is None:
				raise RuntimeError("missing title")
			self.description = grid.procVariables(obj.get("description"))
			valuesObj = obj.get("values")
			if valuesObj is None:
				raise RuntimeError("missing values")
			elif isinstance(valuesObj, str):
				self.buildFromListStr(id, grid, valuesObj)
			else:
				for key, val in valuesObj.items():
					self.values.append(AxisValue(self, grid, key, val))

	def buildFromListStr(self, id, grid, listStr):
		isSplitByDoublePipe = "||" in listStr
		valueList = listStr.split("||" if isSplitByDoublePipe else ",")
		mode = validModes.get(cleanModeName(str(id)))
		if mode is None:
			raise RuntimeError(f"Invalid axis '{mode}': unknown mode")
		if mode.type == "integer":
			valueList = expandNumericListRanges(valueList, int)
		elif mode.type == "decimal":
			valueList = expandNumericListRanges(valueList, float)
		index = 0
		for val in valueList:
			#try:
			val = str(val).strip()
			index += 1
			if isSplitByDoublePipe and val == "" and index == len(valueList):
				continue
			self.values.append(AxisValue(self, grid, str(index), f"{id}={val}"))
			#except Exception as e:
			#	raise RuntimeError(f"value '{val}' errored: {e}")

class GridFileHelper:
	def procVariables(self, text) -> str | None:
		if text is None:
			return None
		text = str(text)
		for key, val in self.variables.items():
			text = text.replace(key, val)
		return text

	def parseYaml(self, yamlContent: dict, gridFile: str):
		self.variables = dict()
		self.axes = list()
		yamlContent = fixDict(yamlContent)
		varsObj = fixDict(yamlContent.get("variables"))
		if varsObj is not None:
			for key, val in varsObj.items():
				self.variables[str(key).lower()] = str(val)
		self.gridObj = fixDict(yamlContent.get("grid"))
		if self.gridObj is None:
			raise RuntimeError(f"Invalid file {gridFile}: missing basic 'grid' root key")
		self.title = self.gridObj.get("title")
		self.description = self.gridObj.get("description")
		self.author = self.gridObj.get("author")
		self.format = self.gridObj.get("format")
		self.OutPath = self.gridObj.get("outpath")
		self.footer = self.gridObj.get("footer") or 'Images area auto-generated by an AI (Stable Diffusion) and so may not have been reviewed by the page author before publishing.\n<script src="a1111webui.js?vary=9"></script>'
		DataLog(f"saving to {self.OutPath}", True, 1)
		if self.title is None or self.description is None or self.author is None or self.format is None:
			raise RuntimeError(f"Invalid file {gridFile}: missing grid title, author, format, or description in grid obj {self.gridObj}")
		self.params = fixDict(self.gridObj.get("params"))
		axesObj = fixDict(yamlContent.get("axes"))
		if axesObj is None:
			raise RuntimeError(f"Invalid file {gridFile}: missing basic 'axes' root key")
		for id, axisObj in axesObj.items():
			self.axes.append(Axis(self, id, axisObj if isinstance(axisObj, str) else fixDict(axisObj)))
		totalCount = 1
		for axis in self.axes:
			totalCount *= len(axis.values)
		if totalCount <= 0:
			raise RuntimeError(f"Invalid file {gridFile}: something went wrong ... is an axis empty? total count is {totalCount} for {len(self.axes)} axes")
		cleanDesc = self.description.replace('\n', ' ')
		print(f"Loaded grid file, title '{self.title}', description '{cleanDesc}', with {len(self.axes)} axes... combines to {totalCount} total images")
		return self

######################### Actual Execution Logic #########################

class SingleGridCall:
	def __init__(self, values: list) -> None:
		self.values = values
		self.skip = False
		skipDict = {'title': [], 'params': []}
		titles = []
		params = []
		for val in values:
			if val.skip:
				self.skip = True
			if hasattr(val, 'skipList') and isinstance(val.skipList, dict):
				if 'title' in val.skipList.keys():
					skipDict['title'] = skipDict['title'] + val.skipList['title']
				if 'params' in val.skipList.keys():
					skipDict['params'] = skipDict['params'] + val.skipList['params']
				
			if hasattr(val, 'title'):
				titles.append(str(val.title))
			if hasattr(val, 'params'):
				params.append(str(val.params))
		skipTitle = skipDict['title']
		skipParams = skipDict['params']
		if skipTitle is not None:
			for item in skipTitle:
				item = str(item).lower()
				if item in map(str.lower, str(titles)):
					self.skip = True
		if skipParams is not None:
			for item in skipParams:
				if any(item in string for string in str(params).lower()):
					self.skip = True
		if gridCallInitHook is not None:
			gridCallInitHook(self)

	def flattenParams(self) -> None:
		global grid
		self.params = grid.params.copy() if grid.params is not None else dict()
		for val in self.values:
			for p, v in val.params.items():
				if gridCallParamAddHook is None or not gridCallParamAddHook(self, grid, p, v):
					self.params[p] = v
	
	@QuickCache(maxSize=100000, localCache="./cache/apply", periodicTime=10)
	def applyTo(self, p: StableDiffusionProcessing):
		for name, val in self.params.items():
			mode = validModes[cleanModeName(name)]
			if name == cleanModeName("Temp Mixed Model"):
				MixModels.append(id(p))

			mode.apply(p, val)

		if gridCallApplyHook is not None:
			gridCallApplyHook(self, p)


class GridRunner:
	def __init__(self, doOverwrite: bool, basePath: str, promptskey: StableDiffusionProcessing):
		global grid
		self.grid: GridFileHelper = grid
		self.totalRun:int = 0
		self.totalSkip:int = 0
		self.totalSteps:int = 0
		self.doOverwrite: bool = doOverwrite
		self.basePath: str = basePath
		self.promptskey: StableDiffusionProcessing	 = promptskey
		self.appliedSets = {}
		grid.minWidth = None
		grid.minHeight = None
		grid.initialPromptskey = promptskey
		self.lastUpdate = []
		
	def updateLiveFile(self, newFile: str):
		tNow: datetime = datetime.datetime.now()
		self.lastUpdate = [x for x in self.lastUpdate if tNow - x['t'] < 20]
		self.lastUpdate.append({'f': newFile, 't': tNow})
		with open(file=os.path.join(self.basepath, 'last.js'), mode='w', encoding="utf-8") as f:
			updateStr = '", "'.join([x['f'] for x in self.lastUpdate])
			f.write(f'window.lastUpdated = ["{updateStr}"]')

	def buildValueSetList(self, axisList: list) -> list:
		# Convert the mutable list to a tuple
		axisTuple = tuple(axisList)
		
		# Define a helper function to perform the actual computation
		@QuickCache(maxSize=1000, localCache="./cache/ValueSets")
		def _buildValueSetList(axisTuple):
			result = list()
			if len(axisTuple) == 0:
				return result
			curAxis = axisTuple[0]
			if len(axisTuple) == 1:
				for val in curAxis.values:
					if not val.skip:
						newList = list()
						newList.append(val)
						temp = SingleGridCall(newList)
						if not temp.skip:
							result.append(temp)
				return result
			nextAxisList = axisTuple[1::]
			for obj in _buildValueSetList(nextAxisList):
				for val in curAxis.values:
					if not val.skip:
						newList = obj.values.copy()
						newList.append(val)
						temp = SingleGridCall(newList)
						if not temp.skip:
							result.append(temp)
			return result

		# Call the helper function with the tuple argument
		return _buildValueSetList(axisTuple)

	def preprocess(self):
		self.valueSetsTemp = self.buildValueSetList(list(reversed(self.grid.axes)))
		self.valueSets = []
		DataLog(f'Have {len(self.valueSetsTemp)} unique value sets, will go into {self.basePath}', True, 1)
		for set in self.valueSetsTemp:
			set.filepath = os.path.join(self.basePath, *map(lambda v: v.path, set.values))
			set.data = ', '.join(list(map(lambda v: f"{v.axis.title}={v.title}", set.values)))
			set.flattenParams()
			if set.skip or (not self.doOverwrite and os.path.exists(os.path.join(set.filepath))):
				self.totalSkip += 1
			else:
				self.totalRun += 1
				self.valueSets.append(set)

	def calculateSteps(self, processors):
		for item in processors:
			steps = 0
			steptemp = []
			def getAll(li:list)-> list:
				items = []
				for anitem in li:
					if isinstance(anitem, list):
						items.extend(getAll(anitem))
					else:
						items.append(anitem)
				return items

			steps += max(DataLog(getAll(item.giga['multistep']), True, 2))
			if hasattr(item, "hr_second_pass_steps"):
				steps += item.hr_second_pass_steps
			self.totalSteps += steps

	def run(self, dry: bool):
		starttime = datetime.datetime.now()
		if gridRunnerPreRunHook is not None:
			gridRunnerPreRunHook(self)
		iteration = 0
		promptBatchList = []
		if not dry:
			threads = []
			for set in self.valueSets:
				if set.skip:
					continue
				iteration += 1
				#if not dry:
				DataLog(f'[mainRun] On {iteration}/{self.totalRun} ... Set: {set.data}, file {set.filepath}', False, 1) #excessive logging causes slowdowns
				p2 = copy(self.promptskey)
				set.applyTo(p2)
				promptBatchList.append(p2)
				#self.appliedSets.add()
				self.appliedSets[id(p2)] = self.appliedSets.get(id(p2), []) + [set]
			DataLog(f'[mainRun] setup phase completed in {(datetime.datetime.now() - starttime).total_seconds():.2f}. batching now', True, 1)
			DataLog(f'[mainRun]\ttotal time\tbatch size\ttime per image\ttime per image step\t sampler name\theight\twidth', False, 0)
			promptBatchList = self.findStepUp(processors=promptBatchList)
			promptBatchList = self.batchPromptsGrouping(promptBatchList)
			self.calculateSteps(promptBatchList)
			stepsremaining = self.totalSteps
			for i, p2 in enumerate(promptBatchList):
				#appliedsets = self.appliedSets[id(p2)]
				p2.total_steps = self.totalSteps
				stepsremaining = stepsremaining - p2.steps
				p2.giga['format'] = grid.format
				p2.giga['start'] = datetime.datetime.now()
				#print(f'On {i+1}/{len(promptbatchlist)} ... Prompts: {p2.prompt[0]}')
				#p2 = StableDiffusionProcessing(p2)
				start2 = datetime.datetime.now()
				DataLog(f"[mainRun] start time: {start2}", True, 2)
				if gridRunnerPreDryHook is not None:
					gridRunnerPreDryHook(self)
				last = gridRunnerRunPostDryHook(self, p2)

				def saveOffThread():
					#aset = copy(appliedsets)
					last2:Processed = copy(last)
					p3:StableDiffusionProcessing = copy(x=p2)
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
					savePaths = getCurrentStepSavepath(p3, max(p3.giga['multistep']))
					if 'inf_grid_use_result_index' in p3.giga:
						iterator = p3.giga['inf_grid_use_result_index']
						path = savePaths[iterator]
						prompt = p3.prompt
						seed = p3.seed
						info = processing.create_infotext(p3, [prompt], [seed], [p3.subseed], [])

						#set = list(aset)[iterator]
						img.gigauncomp['savePath'][iterator]
						#print(f"saving to {path}")
						images.save_image(img, path=os.path.dirname(path), basename="",
							forced_filename=os.path.basename(path), save_to_dirs=False,
							extension=grid.format, p=p3, prompt=prompt,seed=seed, info=info)
					#print(p3.gigauncomp['savePath'])
					for iterator, img in enumerate(last2.images):
						#set = list(aset)[iterator]
						path = savePaths[iterator]
						#print(path)
						DataLog(f"saving to {path}", True, 1)
						images.save_image(img, path=os.path.dirname(path), basename="",
							forced_filename=os.path.basename(path), save_to_dirs=False,
							extension=grid.format, p=p3, prompt=p3.prompt[iterator],seed=last2.seed, 
							info=processing.create_infotext(p3, p3.prompt, p3.seed,
									    p3.subseed, f"Time taken: {quickTimeMath(start2)}", p3.iteration, iterator))
						#self.updateLiveFile(path)
				threading.Thread(target=saveOffThread).start()
				end2 = datetime.datetime.now()
				steptotal = (end2 - start2).total_seconds()
				print(f'the last batch took {steptotal:.2f} for {p2.batch_size} images. an average generation speed of {steptotal / p2.batch_size} per image, and {steptotal / p2.batch_size / p2.steps} seconds per image step')
				print(f"steps remaining: {stepsremaining} / {steptotal}")
				DataLog(f'{steptotal:.2f}\t{p2.batch_size}\t{steptotal / p2.batch_size}\t{steptotal/p2.batch_size/p2.steps}\t{p2.sampler_name}\t{p2.height}\t{p2.width}', False, 0)
		endtime = datetime.datetime.now()
		DataLog(f'Done, Time Taken: {(endtime - starttime).total_seconds():.2f}', True, 2)
		return last
	
	def batchPromptsGrouping(self, Processors: list[StableDiffusionProcessing]) -> list:
		ProcessorGroups = []
		ProcessorGroup = []
		Processor2 = Processors[0]
		for i, processor1 in enumerate(Processors):
			if processor1 == Processor2:
				ProcessorGroup = [processor1]
				continue
			if not self.compareProcessor(processor1, Processor2, ['prompt', 'all_prompts', 'all_negative_prompts', 'negative_prompt', 'seed', 'subseed', 'gigauncomp'], False) or processor1.batch_size == len(ProcessorGroup) - 1:
				ProcessorGroups.append(ProcessorGroup)
				ProcessorGroup = [processor1]
				Processor2 = processor1
			else:
				ProcessorGroup.append(DataLog(processor1, False, 2))
				Processor2 = processor1
		if ProcessorGroup:
			ProcessorGroups.append(ProcessorGroup)

		#print(f"First prompt: {ProcessorGroups[0][0].prompt}")
		print("added all to groups")
		mergedPrompts = []
		print(f"To make use of batching, there will be {len(ProcessorGroups)} batches generated")
		for promgroup in ProcessorGroups:
			first = promgroup[0]

			mergedPrompt:StableDiffusionProcessing = copy(first)

			#print(f"merged prompt pre: {mergedPrompt.prompt}")
			mergedPrompt.prompt: list[str] = []
			#print(f"merged prompt post: {mergedPrompt.prompt}")
			mergedPrompt.negative_prompt: list[str] = []
			mergedPrompt.seed: list[int] = []
			mergedPrompt.subseed: list[int] = []
			#mergedPrompt.giga = {}
			mergedPrompt.gigauncomp = {}

			#print(f"there are {len(promgroup)} merged Processors")
			for i, processor in enumerate(promgroup):
				mergedPrompt.prompt.append(processor.prompt)
				#print(f"merged prompt new: {mergedPrompt.prompt}")
				mergedPrompt.negative_prompt.append(processor.negative_prompt)
				mergedPrompt.seed.append(processor.seed)
				mergedPrompt.subseed.append(processor.subseed)

				for item in processor.gigauncomp:
					#print(f"giga item: {item}")
					try:
						if isinstance(processor.gigauncomp[item], list):
							mergedPrompt.gigauncomp[item].extend(processor.gigauncomp[item])
						else:
							mergedPrompt.gigauncomp[item].append(processor.gigauncomp[item])
					except:
						mergedPrompt.gigauncomp[item] = processor.gigauncomp[item]

			mergedPrompt.batch_size = len(promgroup)
			mergedPrompts.append(mergedPrompt)

		print(f"there are {len(mergedPrompts)} generations after merging")
		return mergedPrompts
	
	def preGroup(self, processors: list[StableDiffusionProcessing]) -> dict:
		mergedDict: dict = {}

		for processor in processors:
			model = processor.override_settings['sd_model_checkpoint']
			sampler = processor.sampler_name

			# Combine the model and sampler names to create the key
			key = f"{sampler}_{model}"

			if key in mergedDict:
				mergedDict[key].append(processor)
			else:
				mergedDict[key] = [processor]

		return mergedDict

	def findStepUp(self, processors):
		start1 = datetime.datetime.now()
		print(f"starting at: {start1}")
		stepProcessor = []  # List to store the modified processors
		print(f"There will be {len(processors)} images saved")
		processorsMerged = self.preGroup(processors)

		# Create a copy of processors to avoid modifying the original list
		for processors in processorsMerged.values():
			remaining_processors = processors[:]
			removed = []
			#with concurrent.futures.ThreadPoolExecutor() as executor:
			for processor1 in processors:
				processor = copy(processor1)  # Create a deep copy
				if processor1 in removed: continue
				stepgroup = [(processor1, processor1.steps, self.appliedSets[id(processor1)][0])]

				to_remove = [processor1]  # Processors to remove from remaining_processors
				for processor2 in remaining_processors:
					if processor2 in to_remove: continue
					if 1 == 2: # self.compareProcessor(processor1, processor2, ['steps', 'giga']):
						stepgroup.append((processor2, processor2.steps, self.appliedSets[id(processor2)][0]))
						if processor.steps < processor2.steps:
							processor.steps = processor2.steps
						to_remove.append(processor2)
				for processor in to_remove:
					removed.extend(to_remove)
					if processor in remaining_processors:
						remaining_processors.remove(processor)

				setattr(processor, 'giga', {})
				setattr(processor, 'gigauncomp', {})
				processor.giga['multistep'] = [item[1] for item in stepgroup]
				processor.steps = max(processor.giga['multistep'])
				#processor.gigauncomp['simpleUpscaleH'] = {item[2].key: [item[2].value] for item in mheight}
				#processor.gigauncomp['simpleUpscaleW'] = {item[2].key: [item[2].value] for item in mwidth}
				processor.gigauncomp['savePath'] = [item[2].filepath for item in stepgroup]
				processor.gigauncomp['appliedSet'] = [item[2] for item in stepgroup]
				stepProcessor.append(processor)

		DataLog(f"To prevent reworking, there will be {len(stepProcessor)} images generated, saving at various step counts between them.", True, 1)
		end1 = datetime.datetime.now()
		DataLog(f"ending at: {end1}, total time: {quickTimeMath(start1, end1)}", True, 1)

		return stepProcessor
	
	#@QuickCache(5000, "./cache/compProc", 60)
	def compareProcessor(self, processor1, processor2, validList, doPrint=False) -> bool:
		#first check to make sure all the attributes exist in both
		#DataLog(f"testing for these: {validList}", doprint=doPrint, level=2)
		if processor1 == processor2: return False
		if not all(hasattr(processor2, attr) for attr in dir(processor1)):
			#DataLog(f"prompt missing attribute {str(attr)}, merge unavailable.", doPrint, 0)
			return False
		for attr in dir(processor1):
			#ignore callables and built in and similar
			if attr.startswith("__"): continue
			if callable(getattr(processor1, attr)): continue
			if isinstance(getattr(processor1, attr), types.BuiltinFunctionType) or isinstance(getattr(processor1, attr), types.BuiltinMethodType): continue
			if attr in validList: continue
			try:
				if getattr(processor1, attr) == getattr(processor2, attr): continue
				else: return False
			except:
				return False
		#DataLog(f"The test returned true.", doPrint, 2)
		return True
	
def PostHandler(gridRunner: GridRunner, promptkey: StableDiffusionProcessing, appliedsets: dict) -> Processed:
	return
######################### Web Data Builders #########################

class WebDataBuilder():
	def buildJson(grid: GridFileHelper, publishGenMetadata: bool, p, dryrun: bool):
		def GetAxis(axis: str):
			id = grid.gridObj.get(axis)
			if id is None:
				return ''
			id = str(id).lower()
			if id == 'none':
				return 'none'
			possible = [x.id for x in grid.axes if x.rawID == id]
			if len(possible) == 0:
				raise RuntimeError(f"Cannot find axis '{id}' for axis default '{axis}'... valid: {[x.rawID for x in grid.axes]}")
			return possible[0]
		showDescrip = grid.gridObj.get('show descriptions')
		result = {
			'title': grid.title,
			'description': grid.description,
			'ext': grid.format,
			'minWidth': grid.minWidth,
			'minHeight': grid.minHeight,
			'defaults': {
				'show_descriptions': True if showDescrip is None else showDescrip,
				'autoscale': grid.gridObj.get('autoscale') or False,
				'sticky': grid.gridObj.get('sticky') or False,
				'x': GetAxis('x axis'),
				'y': GetAxis('y axis'),
				'x2': GetAxis('x super axis'),
				'y2': GetAxis('y super axis')
			}
		}
		if not dryrun:
			result['will_run'] = True
		if publishGenMetadata:
			result['metadata'] = None if webDataGetBaseParamData is None else webDataGetBaseParamData(p)
		axes = list()
		for axis in grid.axes:
			exported = list()
			jAxis = {}
			jAxis['id'] = str(axis.id).lower()
			jAxis['title'] = axis.title
			jAxis['description'] = axis.description or ""
			values = list()
			for val in axis.values:
				if val.path in exported:
					continue
				exported.append(val.path)
				jVal = {}
				jVal['key'] = str(val.key).lower()
				jVal['title'] = val.title
				jVal['description'] = val.description or ""
				jVal['show'] = val.show
				if publishGenMetadata:
					jVal['params'] = val.params
				values.append(jVal)
			jAxis['values'] = values
			axes.append(jAxis)
		result['axes'] = axes
		return json.dumps(result)


	def buildYaml(grid):
		result = {}
		main = {}
		main['title'] = grid.title
		main['description'] = grid.description or ""
		main['format'] = grid.format
		main['author'] = grid.author
		main['footer'] = grid.footer
		result['grid'] = main
		axes = {}
		for axis in grid.axes:
			jAxis = {}
			jAxis['title'] = axis.title
			id = re.sub('__[\d]+$', '', str(axis.id)).replace('_', ' ')
			dups = sum(x == id for x in axes.keys())
			if dups > 0:
				id += (" " * len(dups)) # hack to allow multiples of same id, like for `prompt replace`
				jAxis['title'] += " " + dups
			values = list(map(lambda val: str(val.title), axis.values))
			jAxis['values'] = ' || '.join(values)
			axes[id] = jAxis
		result['axes'] = axes
		return result

	def radioButtonHtml(name, id, descrip, label):
		return f'<input type="radio" class="btn-check" name="{name}" id="{str(id).lower()}" autocomplete="off" checked=""><label class="btn btn-outline-primary" for="{str(id).lower()}" title="{descrip}">{HTMLModule.escape(label)}</label>\n'
	
	def axisBar(label, content):
		return f'<br><div class="btn-group" role="group" aria-label="Basic radio toggle button group">{label}:&nbsp;\n{content}</div>\n'

	def buildHtml(grid):
		with open(os.path.join(AssetDir, 'page.html'), 'r', encoding='utf=8') as referenceHtml:
			html = referenceHtml.read()
		xSelect = ""
		ySelect = ""
		x2Select = WebDataBuilder.radioButtonHtml('x2_axis_selector', f'x2_none', 'None', 'None')
		y2Select = WebDataBuilder.radioButtonHtml('y2_axis_selector', f'y2_none', 'None', 'None')
		content = '<div style="margin: auto; width: fit-content;"><table class="sel_table">\n'
		advancedSettings = ''
		primary = True
		for axis in grid.axes:
			try:
				axisDescrip = cleanForWeb(axis.description or '')
				trClass = "primary" if primary else "secondary"
				content += f'<tr class="{trClass}">\n<td>\n<h4>{HTMLModule.escape(str(axis.title))}</h4>\n'
				advancedSettings += f'\n<h4>{axis.title}</h4><div class="timer_box">Auto cycle every <input style="width:30em;" autocomplete="off" type="range" min="0" max="360" value="0" class="form-range timer_range" id="range_tablist_{axis.id}"><label class="form-check-label" for="range_tablist_{axis.id}" id="label_range_tablist_{axis.id}">0 seconds</label></div>\nShow value: '
				axisClass = "axis_table_cell"
				if len(axisDescrip.strip()) == 0:
					axisClass += " emptytab"
				content += f'<div class="{axisClass}">{axisDescrip}</div></td>\n<td><ul class="nav nav-tabs" role="tablist" id="tablist_{axis.id}">\n'
				primary = not primary
				isFirst = axis.default is None
				exported = []
				for val in axis.values:
					if val.path in exported:
						continue
					exported.append(val.path)
					if axis.default is not None:
						isFirst = str(axis.default) == str(val.key)
					selected = "true" if isFirst else "false"
					active = " active" if isFirst else ""
					isFirst = False
					descrip = cleanForWeb(val.description or '')
					content += f'<li class="nav-item" role="presentation"><a class="nav-link{active}" data-bs-toggle="tab" href="#tab_{axis.id}__{val.key}" id="clicktab_{axis.id}__{val.key}" aria-selected="{selected}" role="tab" title="{HTMLModule.escape(str(val.title))}: {descrip}">{HTMLModule.escape(str(val.title))}</a></li>\n'
					advancedSettings += f'&nbsp;<input class="form-check-input" type="checkbox" autocomplete="off" id="showval_{axis.id}__{val.key}" checked="true" onchange="javascript:toggleShowVal(\'{axis.id}\', \'{val.key}\')"> <label class="form-check-label" for="showval_{axis.id}__{val.key}" title="Uncheck this to hide \'{HTMLModule.escape(str(val.title))}\' from the page.">{HTMLModule.escape(str(val.title))}</label>'
				advancedSettings += f'&nbsp;&nbsp;<button class="submit" onclick="javascript:toggleShowAllAxis(\'{axis.id}\')">Toggle All</button>'
				content += '</ul>\n<div class="tab-content">\n'
				isFirst = axis.default is None
				for val in axis.values:
					if axis.default is not None:
						isFirst = str(axis.default) == str(val.key)
					active = " active show" if isFirst else ""
					isFirst = False
					descrip = cleanForWeb(val.description or '')
					if len(descrip.strip()) == 0:
						active += " emptytab"
					content += f'<div class="tab-pane{active}" id="tab_{axis.id}__{val.key}" role="tabpanel"><div class="tabval_subdiv">{descrip}</div></div>\n'
			except Exception as e:
				raise RuntimeError(f"Failed to build HTML for axis '{axis.id}': {e}")
			content += '</div></td></tr>\n'
			xSelect += WebDataBuilder.radioButtonHtml('x_axis_selector', f'x_{axis.id}', axisDescrip, axis.title)
			ySelect += WebDataBuilder.radioButtonHtml('y_axis_selector', f'y_{axis.id}', axisDescrip, axis.title)
			x2Select += WebDataBuilder.radioButtonHtml('x2_axis_selector', f'x2_{axis.id}', axisDescrip, axis.title)
			y2Select += WebDataBuilder.radioButtonHtml('y2_axis_selector', f'y2_{axis.id}', axisDescrip, axis.title)
		content += '</table>\n<div class="axis_selectors">'
		content += WebDataBuilder.axisBar('X Axis', xSelect)
		content += WebDataBuilder.axisBar('Y Axis', ySelect)
		content += WebDataBuilder.axisBar('X Super-Axis', x2Select)
		content += WebDataBuilder.axisBar('Y Super-Axis', y2Select)
		content += '</div></div>\n'
		html = html.replace("{TITLE}", grid.title).replace("{CLEAN_DESCRIPTION}", cleanForWeb(grid.description)).replace("{DESCRIPTION}", grid.description).replace("{CONTENT}", content).replace("{ADVANCED_SETTINGS}", advancedSettings).replace("{AUTHOR}", grid.author).replace("{EXTRA_FOOTER}", grid.footer).replace("{VERSION}", getVersion())
		return html

	def EmitWebData(path, publish_gen_metadata, p, yamlContent, dryrun: bool):
		global grid
		print("Building final web data...")
		os.makedirs(path, exist_ok=True)
		json = WebDataBuilder.buildJson(grid, publish_gen_metadata, p, dryrun)
		if not dryrun:
			with open(os.path.join(path, 'last.js'), 'w+', encoding="utf-8") as f:
				f.write("windows.lastUpdated = []")
		with open(os.path.join(path,"data.js"), 'w', encoding="utf-8") as f:
			f.write("rawData = " + json)
		if yamlContent is None:
			yamlContent = WebDataBuilder.buildYaml(grid)
		with open(os.path.join(path, "config.yml"), 'w', encoding="utf-8") as f:
			yaml.dump(yamlContent, f, sort_keys=False, default_flow_style=False, width=1000)
		for f in ["bootstrap.min.css", "jsgif.js", "bootstrap.bundle.min.js", "proc.js", "jquery.min.js", "styles.css", "placeholder.png"] + ExtraAssets:
			shutil.copyfile(os.path.join(AssetDir, f), os.path.join(path, f))
		html = WebDataBuilder.buildHtml(grid)
		with open(os.path.join(path, "index.html"), 'w', encoding="utf-8") as f:
			f.write(html)
		print(f"Web file is now at {os.path.join(path,'index.html')}")

######################### Main Runner Function #########################

def runGridGen(passThroughObj: StableDiffusionProcessing, inputFile: str, outputFolderBase: str, outputFolderName: str = None, doOverwrite: bool = False, generatePage: bool = True, publishGenMetadata: bool = True, dryRun: bool = False, manualPairs: list = None):
	global grid, logFile, cacheFile, assetFile
	startTime = datetime.datetime.now()
	grid = GridFileHelper()
	yamlContent = None
	assetFile = inputFile
	if manualPairs is None:
		fullInputPath = os.path.realpath(os.path.join(AssetDir, inputFile))
		if not os.path.exists(fullInputPath):
			raise RuntimeError(f"Non-existent file '{inputFile}'")
		# Parse and verify
		with open(fullInputPath, 'r', encoding="utf-8") as yamlContentText:
			try:
				yamlContent = yaml.safe_load(yamlContentText)
			except yaml.YAMLError as exc:
				raise RuntimeError(f"Invalid YAML in file '{inputFile}': {exc}")
			grid.parseYaml(yamlContent, inputFile)
	else:
		grid.title = outputFolderName
		grid.description = ""
		grid.variables = dict()
		grid.author = "Unspecified"
		grid.format = "png"
		grid.axes = list()
		grid.params = None
		for i in range(0, int(len(manualPairs) / 2)):
			key = manualPairs[i * 2]
			if isinstance(key, str) and key.strip() != "":
				try:
					grid.axes.append(Axis(grid, key, manualPairs[i * 2 + 1]))
				except Exception as e:
					raise RuntimeError(f"Invalid axis {(i + 1)} '{key}': errored: {e}")
	pairTime = datetime.datetime.now()
	# Now start using it
	if outputFolderName.strip() == "":
		if grid.OutPath is None:
			outputFolderName = inputFile.replace(".yml", "")
		else:
			outputFolderName = grid.OutPath.strip()
	print("will save to: " + outputFolderName)
	if os.path.isabs(outputFolderName):
		folder = outputFolderName
	else:
		folder = os.path.join(outputFolderBase, outputFolderName)
	runner = GridRunner(doOverwrite, folder, passThroughObj)
	logFile = DataLogSetup(folder)
	cacheFile = os.path.join(folder, "cache")
	runner.preprocess()
	prerun = datetime.datetime.now()
	if generatePage:
		json = WebDataBuilder.EmitWebData(folder, publishGenMetadata, passThroughObj, yamlContent, dryRun)
	runnerStart = datetime.datetime.now()
	DataLog(f"Main runner started at {startTime}, pairing done at {pairTime}, preprocess at {prerun}, and processor started at {runnerStart} time for each: {quickTimeMath(startTime=startTime, endTime=pairTime)}, {quickTimeMath(startTime=startTime, endTime=prerun)}, {quickTimeMath(startTime=startTime, endTime=runnerStart)}", True, 1)
	result = runner.run(dryRun)
	if dryRun:
		print("Infinite Grid dry run succeeded without error")
	#else:
	#	json = str(json).replace('"will_run": true, ', '')
	#	with open(os.path.join(folder, "data.js"), 'w', encoding="utf-8") as f:
	#		f.write("rawData = " + json)
	#	os.remove(os.path.join(folder, "last.js"))
	return result