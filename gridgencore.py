# This file is part of Infinity Grid Generator, view the README.md at https://github.com/mcmonkeyprojects/sd-infinity-grid-generator-script for more information.

import os, glob, yaml, json, shutil, math, re, threading, hashlib, types, datetime, re, atexit, signal
from multiprocessing import Pool, cpu_count as cpuCount
from modules import sd_models as sdModels, images, processing
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img, Processed
from modules.shared import opts
from copy import copy, deepcopy
from PIL import Image
from git import Repo
from colorama import init as CInit, Fore, Style
from functools import lru_cache, wraps
import pickle
CInit()

######################### Core Variables #########################

AssetDir = os.path.dirname(__file__) + "/assets"
ExtraFooter = "..."
ExtraAssets = []
validModes = {}
ImagesCache = None
modelchange = {}
logFile: str
cacheFile: str = None
Version:str = '23.9.5'
printLevel: int= 1
grid = None
quickListCache:dict = {}
paramcache = []
lock = threading.Lock()
cleanList: dict = {}

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

class quickCache:
	def __init__(self,maxsize:int,localcache, periodicTime=600):
		global cacheFile
		self.maxSize = 1000
		self.cache = {}
		self.inited = False
		
		self.lock = threading.Lock()
		try:
			self.loadCache()
		except:
			pass
		#self.periodicCacheWrite()

	
	def __call__(self,maxsize:int,localcache, periodicTime=600) ->callable:
		if not self.inited:
			if localcache is not None:
				self.cacheFile = localcache
			else: self.cacheFile = cacheFile
			print(f"{self.cacheFile}, {localcache}" )
			self.cachePath = os.path.dirname(os.path.realpath(self.cacheFile))
			if not os.path.exists(self.cachePath): os.makedirs(self.cachePath) 
			self.backupFile = self.cacheFile + "." + "bak"
			self.maxSize = maxsize
			self.inited = True
			self.loadCache()
			self.periodicCacheWrite(periodicTime)
		return self.decorator(maxsize)

	def loadCache(self):
		if os.path.exists(self.cacheFile):
			try:
				with open(self.cacheFile, 'rb') as file:
					self.cache = pickle.load(file)
			except Exception as e:
				print(f"failed to load cache: {e}")

	def saveCache(self):
		with self.lock:
			try:
				with open(self.cacheFile, 'rb') as file:
					tempCache = pickle.load(file)
				tempCache.update(self.cache)
			except:
				tempCache = self.cache
			with open(self.cacheFile, 'wb') as file:
				pickle.dump(tempCache,file)
			with open(self.backupFile, 'wb') as backup:
				pickle.dump(tempCache,backup)

	
	def periodicCacheWrite(self, periodicTime):
		interval = periodicTime
		threading.Timer(interval, self.periodicCacheWrite).start()
		self.saveCache()

	def decorator(self, func):
		@wraps(func)
		def wrapper(*args, **kwargs):
			keyArgs = []
			for arg in args:
				if isinstance(arg, StableDiffusionProcessing):
					keyArgs.append(hashlib.blake2s(str(arg).encode()).hexdigest())
				else: 
					keyArgs.append(arg)

			key = (tuple(keyArgs), frozenset(kwargs.items()))
			if key in self.cache:
				return self.cache[key]
			result = func(*args, **kwargs)
			if len(self.cache) >= self.maxsize:
				# Remove the oldest entry to make room for a new one.
				self.cache.pop(next(iter(self.cache)))
			self.cache[key] = result
			return result

#@quickCache(maxsize=1000)
def escapeHTML(text: str) -> str:
	return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

def dataLog(message: str, doprint: bool, level: int) -> None:
	try:
		if not os.path.exists(logFile):
			open(logFile, 'x')
		with open(logFile, 'a+', encoding="utf-8") as f:
			f.write(message)
			f.write('\n')
	except: print(f"[datalog] used before defining logFile, value: {message}")
	
	if print: # and printLevel >= level:
		if level == 0: #warning
			print(Fore.RED + message + Style.RESET_ALL)
		elif level == 1: #info
			print(Fore.GREEN + message + Style.RESET_ALL)
		elif level == 2: #debug
			print(Fore.BLUE + message + Style.RESET_ALL)
	
def datalogNew(folder):
	logFile = os.path.join(folder, 'log.txt')

	# Create a list of existing log files in the folder
	logFiles = [logFile] + [os.path.join(folder, f'log{i}.txt') for i in range(1, 5)]

	# Rename existing log files in reverse order
	for i in range(5, 1, -1):
		src = os.path.join(folder, f'log{i - 1}.txt')
		dest = os.path.join(folder, f'log{i}.txt')
		if os.path.exists(src):
			os.rename(src, dest)

	# Rename the current log file to 'log1.txt'
	if os.path.exists(logFile):
		os.rename(logFile, os.path.join(folder, 'log1.txt'))

	return logFile

def quickTimeMath(startTime:datetime.datetime, endTime:datetime.datetime = None) -> str:
	if endTime == None: endTime = datetime.datetime.now()
	return "{:.2f}".format((endTime - startTime).total_seconds())

def lateapplyModel(p, v) -> None:
	starttime = datetime.datetime.now()
	threading.Thread(target=dataLog(f"[lateApplyModel] Changing models to {v} please wait", True, 1)).start()
	opts.sd_model_checkpoint = getModelFor(v)
	sdModels.reload_model_weights()
	threading.Thread(target=dataLog(f"[lateApplyModel] Changing models to done in {quickTimeMath(starttime, datetime.datetime.now())}", True, 1)).start()	

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

def listImageFiles():
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
	threading.Thread(target=dataLog(f"[listImageFiles] done in {quickTimeMath(startTime=starttime)}", False, 2)).start()
	return ImagesCache


def getNameList() -> list[str]:
	fileList = glob.glob(AssetDir + "/*.yml")
	fileList.extend(glob.glob(opts.outdir_txt2img_grids + "/*.yml"))
	fileList.extend(glob.glob(opts.outdir_grids + "/*.yml"))
	fileList.extend(glob.glob(opts.outdir_img2img_grids + "/*.yml"))
	justFileNames = sorted(list(map(lambda f: os.path.relpath(f, AssetDir), fileList)))
	return justFileNames

def fixDict(d: dict):
	if d is None:
		return None
	if type(d) is not dict:
		raise RuntimeError(f"Value '{d}' is supposed to be submapping but isn't (it's plaintext, a list, or some other incorrect format). Did you typo the formatting?")
	return {str(k).lower(): v for k, v in d.items()}

def cleanForWeb(text: str):
	if text is None:
		return None
	if type(text) is not str:
		raise RuntimeError(f"Value '{text}' is supposed to be text but isn't (it's a datamapping, list, or some other incorrect format). Did you typo the formatting?")
	return text.replace('"', '&quot;')

#@quickCache(maxsize=1000,localcache="./cache/aidCache")
def cleanId(id: str) -> str:
	return re.sub("[^a-z0-9]", "_", id.lower().strip())

@quickCache(1000, "./cache/modeNameCache.y")
def cleanModeName(name: str) -> str:
    return name.lower().replace('[', '').replace(']', '').replace(' ', '').replace('\\', '/').replace('//', '/').strip()
	
@quickCache(1000, "./cache/NameCache.y")
def cleanName(name: str) -> str:
	cleanedName = re.sub(r'\[.*?\]', '', name)
	cleanedName = cleanedName.lower().replace(' ', '').replace('\\', '/').replace('//', '/').strip()
	cleanList[name] = cleanedName
	return cleanedName

@quickCache(maxsize=1000,localcache="./cache/quickList.y")
def getQuickList(list:frozenset):
	global quickListCache
    # Calculate a hash for the input list.
	listHash = hashlib.blake2s(str(list).encode()).hexdigest()

    # Check if the list is already cached, and if so, return it.
	if listHash in quickListCache:
		dataLog(f"[getQuickList] using cache", doprint=False, level=2)
		return quickListCache[listHash]
	else: 
		dataLog(f"[getQuickList] not using cache", doprint=False, level=2)

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

def getBestInList(name, list):
	clean = cleanName(name)
	dataLog(f"[getBestInList] name = {name} clean = {clean}", doprint=False, level=2)

    # Get the quick list for the input list.
	quickList = getQuickList(frozenset(list))

	if clean in quickList:
		return quickList[clean][0]
	else:
		return None

@quickCache(maxsize=1000, localcache="./cache/betterName")
def chooseBetterFileName(rawName: str, fullName: str) -> str:
	starttime = datetime.datetime.now()
	partialName = os.path.splitext(os.path.basename(fullName))[0]
	if '/' in rawName or '\\' in rawName or '.' in rawName or len(rawName) >= len(partialName):
		threading.Thread(target=dataLog(f"[chooseBetterFileName] done in {quickTimeMath(startTime=starttime)}", False, 2)).start()
		return rawName
	threading.Thread(target=dataLog(f"[chooseBetterFileName] done in {quickTimeMath(startTime=starttime)}", False, 2)).start()
	return partialName

@quickCache(maxsize=1000, localcache="./cache/fixNum")
def fixNum(num) -> float | None:
	if num is None or math.isinf(num) or math.isnan(num):
		return None
	return num

@quickCache(maxsize=1000, localcache="./cache/Numeric")
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

def registerMode(name: str, mode: GridSettingMode):
	mode.name = name
	validModes[cleanModeName(name)] = mode

######################### Validation #########################

@quickCache(maxsize=1000)
def validateParams(grid, params: dict):
	global paramcache
	phash = hashlib.blake2s(str(params).encode()).hexdigest()
	if phash in paramcache:
		dataLog("valid", False, 2)
	else:
		for p,v in params.items():
			params[p] = validateSingleParam(p, grid.procVariables(v))
		paramcache.append(phash)

@quickCache(maxsize=1000)
def validateSingleParam(p: str, value):
	p = cleanModeName(p)
	mode = validModes.get(p)
	if mode is None:
		raise RuntimeError(f"Invalid grid parameter '{p}': unknown mode")
	modeType = mode.type
	if modeType == int:
		vInt = int(value)
		if vInt is None:
			raise RuntimeError(f"Invalid parameter '{p}' as '{value}': must be an integer number")
		min = mode.min
		max = mode.max
		if min is not None and vInt < min:
			raise RuntimeError(f"Invalid parameter '{p}' as '{value}': must be at least {min}")
		if max is not None and vInt > max:
			raise RuntimeError(f"Invalid parameter '{p}' as '{value}': must not exceed {max}")
		value = vInt
	elif modeType == float:
		vFloat = float(value)
		if vFloat is None:
			raise RuntimeError(f"Invalid parameter '{p}' as '{value}': must be a decimal number")
		min = mode.min
		max = mode.max
		if min is not None and vFloat < min:
			raise RuntimeError(f"Invalid parameter '{p}' as '{value}': must be at least {min}")
		if max is not None and vFloat > max:
			raise RuntimeError(f"Invalid parameter '{p}' as '{value}': must not exceed {max}")
		value = vFloat
	elif modeType == bool:
		vClean = str(value).lower().strip()
		if vClean == "true":
			retvalue = True
		elif vClean == "false":
			retvalue = False
		else:
			raise RuntimeError(f"Invalid parameter '{p}' as '{value}': must be either 'true' or 'false'")
	elif modeType == str and mode.validList is not None:
		validList = mode.validList()
		value = getBestInList(value, validList)
		if value is None:
			raise RuntimeError(f"Invalid parameter '{p}' as '{value}': not matched to any entry in list {list(validList)}")
	if mode.clean is not None:
		return mode.clean(p, value)
	return value

def applyField(name: str):
	def applier(p, v):
		setattr(p, name, v)
	return applier

def applyFieldAsImageData(name: str):
	def applier(p, v):
		fName = getBestInList(v, listImageFiles())
		if fName is None:
			raise RuntimeError("Invalid parameter '{p}' as '{v}': image file does not exist")
		path = AssetDir + "/images/" + fName
		image = Image.open(path)
		setattr(p, name, image)
	return applier


######################### YAML Parsing and Processing #########################

class AxisValue:
	def __init__(self, axis, grid, key: str, val):
		self.axis = axis
		self.key = cleanId(str(key))
		if any(x.key == self.key for x in axis.values):
			self.key += f"__{len(axis.values)}"
		self.params = list()
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
			self.title = grid.procVariables(val.get("title"))
			self.description = grid.procVariables(val.get("description"))
			self.skip = False
			self.skipList = val.get("skip")
			if isinstance(self.skipList, bool):
				self.skip = self.skipList
			elif self.skipList is not None and isinstance(self.skipList, dict):
				self.skip = self.skipList.get("always")
			self.params = fixDict(val.get("params"))
			self.show = (str(grid.procVariables(val.get("show")))).lower() != "false"
			if self.title is None or self.params is None:
				raise RuntimeError(f"Invalid value '{key}': '{val}': missing title or params")
			if not self.skip:
				threading.Thread(validateParams(grid, self.params)).start()
	
	def __str__(self):
		return f"(title={self.title}, description={self.description}, params={self.params})"
	def __unicode__(self):
		return self.__str__()

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
		threading.Thread(target=dataLog(f"saving to {self.OutPath}", True, 1)).start()
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
				titles.append(val.title)
			if hasattr(val, 'params'):
				params.append(val.params)
		skipTitle = skipDict['title']
		skipParams = skipDict['params']
		if skipTitle is not None:
			for item in skipTitle:
				if item in map(str.lower, titles):
					self.skip = True
		if skipParams is not None:
			for item in skipParams:
				if any(item in string for string in str(params).lower()):
					self.skip = True
		if gridCallInitHook is not None:
			gridCallInitHook(self)

	def flattenParams(self):
		global grid
		self.params = grid.params.copy() if grid.params is not None else dict()
		for val in self.values:
			for p, v in val.params.items():
				if gridCallParamAddHook is None or not gridCallParamAddHook(self, grid, p, v):
					self.params[p] = v
	
	@quickCache(maxsize=100000, oldCache=cacheFile)
	def applyTo(self, p: StableDiffusionProcessing, dry: bool):
		for name, val in self.params.items():
			mode = validModes[cleanModeName(name)]
			#if not dry or mode.dry:
			if mode == "model" or mode == "Model" or name == "model":
				modelchange[id(p)] = val
			else:
				#startTime = datetime.datetime.now()
				mode.apply(p, val)
				#dataLog(f"[applyTo] Applying {name} took {(datetime.datetime.now() - startTime).total_seconds():.9f}", False, 0)
		if gridCallApplyHook is not None:
			gridCallApplyHook(self, p, dry)


class GridRunner:
	def __init__(self, doOverwrite: bool, basePath: str, promptskey: StableDiffusionProcessing):
		global grid
		self.grid = grid
		self.totalRun = 0
		self.totalSkip = 0
		self.totalSteps = 0
		self.doOverwrite = doOverwrite
		self.basePath = basePath
		self.promptskey = promptskey
		self.appliedSets = {}
		grid.minWidth = None
		grid.minHeight = None
		grid.initialPromptskey = promptskey
		self.lastUpdate = []
		
	def updateLiveFile(self, newFile: str):
		tNow = datetime.datetime.now()
		self.lastUpdate = [x for x in self.lastUpdate if tNow - x['t'] < 20]
		self.lastUpdate.append({'f': newFile, 't': tNow})
		with open(os.path.join(self.basepath, 'last.js'), 'w', encoding="utf-8") as f:
			updateStr = '", "'.join([x['f'] for x in self.lastUpdate])
			f.write(f'window.lastUpdated = ["{updateStr}"]')

	def buildValueSetList(self, axisList: list) -> list:
		# Convert the mutable list to a tuple
		axisTuple = tuple(axisList)
		
		# Define a helper function to perform the actual computation
		@quickCache(maxsize=1000)
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
		threading.Thread(target=dataLog(f'Have {len(self.valueSetsTemp)} unique value sets, will go into {self.basePath}', True, 1)).start()
		for set in self.valueSetsTemp:
			set.filepath = self.basePath + '/' + '/'.join(list(map(lambda v: cleanName(v.key), set.values)))
			set.data = ', '.join(list(map(lambda v: f"{v.axis.title}={v.title}", set.values)))
			set.flattenParams()
			if set.skip:
				self.totalSkip += 1
			elif not self.doOverwrite and os.path.exists(os.path.join(set.filepath + "." + self.grid.format)):
				self.totalSkip += 1
			else:
				self.totalRun += 1
				self.valueSets.append(set)
				stepCount = self.promptskey.steps
				self.totalSteps += stepCount
				enableHR = set.params.get("enable highres fix")
				if enableHR is None:
					enableHR = self.promptskey.enable_hr
				if enableHR:
					highresSteps = set.params.get("highres steps")
					highresSteps = int(highresSteps) if highresSteps is not None else (self.promptskey.hr_second_pass_steps or stepCount)
					self.totalSteps += highresSteps
				
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
				dataLog(f'[mainRun] On {iteration}/{self.totalRun} ... Set: {set.data}, file {set.filepath}', True, 1)
				p2 = copy(self.promptskey)
				set.applyTo(p2, dry)
				promptBatchList.append(p2)
				#self.appliedSets.add()
				self.appliedSets[id(p2)] = self.appliedSets.get(id(p2), []) + [set]
			dataLog(f'[mainRun] setup phase completed in {(datetime.datetime.now() - starttime).total_seconds():.2f}. batching now', True, 1)
			dataLog(f'[mainRun]\ttotal time\tbatch size\ttime per image\ttime per image step\t sampler name\theight\twidth', False, 0)
			promptBatchList = self.batchPrompts(promptBatchList, self.promptskey)
			for i, p2 in enumerate(promptBatchList):
				appliedsets = self.appliedSets[id(p2)]
				#print(f'On {i+1}/{len(promptbatchlist)} ... Prompts: {p2.prompt[0]}')
				#p2 = StableDiffusionProcessing(p2)
				start2 = datetime.datetime.now()
				threading.Thread(target=dataLog(f"[mainRun] start time: {start2}", True, 2)).start()
				if id(p2) in modelchange.keys():
					lateapplyModel(p2,modelchange[id(p2)])
				if gridRunnerPreDryHook is not None:
					gridRunnerPreDryHook(self)
				try:
					last = gridRunnerRunPostDryHook(self, p2, appliedsets)
					#self.updateLiveFile(set.filepath + "." + self.grid.format)
				except Exception as e: 
					threading.Thread(target=dataLog("[mainRun] image failed to generate. please restart later", True, 0)).start()
					threading.Thread(target=dataLog(f"[mainRun] exception: {e}", False, 0)).start()
					continue
				try:
					if p2.hasPostProcessing:
						try:
							PostHandler(self, p2, self.appliedSets[id(p2)])
							print("Post Processing")
						except Exception as e:
							print("Image postprocessing has failed.")
							print(f'error is: {e}' )
				except:
					print("no post processing")
					

				def saveOffThread():
					aset = deepcopy(appliedsets)
					for iterator, img in enumerate(last.images):
						set = list(aset)[iterator]
						threading.Thread(target=dataLog(f"saving to {set.filepath}", True, 1)).start()
						images.save_image(img, path=os.path.dirname(set.filepath), basename="",
							forced_filename=os.path.basename(set.filepath), save_to_dirs=False,
							extension=grid.format, p=p2, prompt=p2.prompt[iterator],seed=last.seed)
				threading.Thread(target=saveOffThread).start()
				#def saveOffThread():
				#	try:
				#		for iterator, img in enumerate(last.images):
				#			set = list(appliedsets)[iterator]
				#			print(f"saving to: {os.path.dirname(set.filepath)}\\{os.path.basename(set.filepath)}")
				#			images.save_image(img, path=os.path.dirname(set.filepath), basename="",
				#				forced_filename=os.path.basename(set.filepath), save_to_dirs=False,
				#				extension=grid.format, p=p2, prompt=p2.prompt[iterator],seed=last.seed)
				#	except FileNotFoundError as e:
				#		if e.strerror == 'The filename or extension is too long' and hasattr(e, 'winerror') and e.winerror == 206:
				#			print(f"\n\n\nOS Error: {e.strerror} - see this article to fix that: https://www.autodesk.com/support/technical/article/caas/sfdcarticles/sfdcarticles/The-Windows-10-default-path-length-limitation-MAX-PATH-is-256-characters.html \n\n\n")
				#		raise e
				#threading.Thread(target=saveOffThread).start()
				end2 = datetime.datetime.now()
				steptotal = (end2 - start2).total_seconds()
				print(f'the last batch took {steptotal:.2f} for {p2.batch_size} images. an average generation speed of {steptotal / p2.batch_size} per image, and {steptotal / p2.batch_size / p2.steps} seconds per image step')
				threading.Thread(target=dataLog(f'{steptotal:.2f}\t{p2.batch_size}\t{steptotal / p2.batch_size}\t{steptotal/p2.batch_size/p2.steps}\t{p2.sampler_name}\t{p2.height}\t{p2.width}', False, 0)).start()
		endtime = datetime.datetime.now()
		threading.Thread(target=dataLog(f'Done, Time Taken: {(endtime - starttime).total_seconds():.2f}', True, 2)).start()
		return last
	
	def batchPromptsGrouping(self, promptList: list, promptKey: StableDiffusionProcessing) -> list:
		# Group prompts by batch size
		prompt_groups = {}
		prompt_group = []
		batchsize = promptKey.batch_size
		starto = 0
		prompt_groups = {}
		prompt_group = []
		starto = 0
		for i in range(len(promptList)):
			prompt = promptList[i]
			if i > 0:
				prompt2 = promptList[i - 1]
			else:
				prompt2 = prompt
			if id(prompt) in modelchange and prompt != prompt2 and id(prompt2) in modelchange and modelchange[id(prompt)] != modelchange[id(prompt2)]:
				if len(prompt_group) > 0:
					prompt_groups[starto] = prompt_group
					starto += 1
				prompt_groups[starto] = prompt
				starto += 1
				prompt_group = []
			elif i % prompt.batch_size == 0:
				if prompt_group:
					prompt_groups[starto] = prompt_group
					starto += 1
					prompt_group = []
				prompt_group.append(prompt)
			else:
				prompt_group.append(prompt)
		if prompt_group:
			prompt_groups[starto] = prompt_group
		print("added all to groups")
		return prompt_groups
	
	def batchPromptsValidating(self, promptGroups: list, promptKey: StableDiffusionProcessing):
		merged_prompts = []
		print(f"there are {len(promptGroups)} groups after grouping. merging now")
		for iterator, promgroup in enumerate(promptGroups):
			promgroup = promptGroups[iterator]
			if isinstance(promgroup, StableDiffusionProcessing) or isinstance(promgroup, int):
				fail = True
			else:
				fail = False
				prompt_attr = promgroup[0]
				batchsize = prompt_attr.batch_size
				print(f"merging prompts {iterator*batchsize} - {iterator*batchsize+batchsize} of {len(promptGroups.items())*batchsize}")

				for it, tempprompt in enumerate(promgroup):
					
					if not all(hasattr(tempprompt2, attr) for tempprompt2 in promgroup for attr in dir(tempprompt)):
						fail = True
						threading.Thread(target=dataLog(f"prompt does not contain {str(attr)} can not merge", False, 0)).start()
						break
					for attr in dir(tempprompt):
						if attr.startswith("__"): continue
						if callable(getattr(tempprompt, attr)): continue
						if isinstance(getattr(tempprompt, attr, None), types.BuiltinFunctionType) or isinstance(getattr(tempprompt, attr, None), types.BuiltinMethodType): continue
						if attr in ['prompt', 'all_prompts', 'all_negative_prompts', 'negative_prompt', 'seed', 'subseed']: continue
						try:
							if getattr(tempprompt, attr) == getattr(prompt_attr, attr): continue
							else: 
								fail = True
								if it == 1: 
									threading.Thread(target=dataLog(f"Prompt contains incorrect {str(attr)} merge unavailable. values are: {str(getattr(tempprompt, attr))}", False, 0)).start()
								threading.Thread(target=dataLog(f"prompt contains incorrect {str(attr)} merge unavailable. values are: {str(getattr(prompt_attr, attr))}", False, 0)).start()
								break
						except AttributeError:
							print(tempprompt)
							threading.Thread(target=dataLog(prompt_attr, False, 0)).start()
							raise
			if not fail:
				merged_prompt = prompt_attr
				merged_prompt.prompt = [p.prompt for p in promgroup]
				merged_prompt.negative_prompt = [p.negative_prompt for p in promgroup]
				merged_prompt.seed = [p.seed for p in promgroup]
				merged_prompt.subseed = [p.subseed for p in promgroup]
				merged_prompts.append(merged_prompt)
				self.totalSteps -= (merged_prompt.batch_size - 1) * merged_prompt.steps
				# Add applied sets
				for prompt in promgroup:
					setup2 = self.appliedSets.get(id(prompt), [])
					#print(setup2)
					merged_filepaths = [setup.filepath for setup in self.appliedSets[id(merged_prompt)]]
					if any(setall.filepath in merged_filepaths for setall in setup2): continue
					if self.appliedSets.get(id(prompt), []) in self.appliedSets[id(merged_prompt)]: continue
					self.appliedSets[id(merged_prompt)] += self.appliedSets.get(id(prompt), [])
				#print("merged")
				merged_prompt.batch_size = len(promgroup)

			if fail and (isinstance(promgroup, StableDiffusionProcessingTxt2Img) or isinstance(promgroup, StableDiffusionProcessing) or isinstance(promgroup, StableDiffusionProcessingImg2Img)):
				promgroup.batch_size = 1
				merged_prompts.append(promgroup)
			elif fail and (isinstance(promgroup, int)):
				continue
			elif fail:
				for prompt in promgroup:
					prompt.batch_size = 1
				merged_prompts.extend(promgroup)
		print(f"there are {len(merged_prompts)} generations after merging")
		return merged_prompts

	def batchPrompts(self, promptList: list, promptKey: StableDiffusionProcessing) -> list:
		return self.batchPromptsValidating(self.batchPromptsGrouping(promptList, promptKey),promptKey)

	#def batchPrompts(self, promptList: list, promptKey: StableDiffusionProcessing)-> list:
	#	promptGroups = []
	#	promptGroup = []
	#	for i in range(len(promptList)):
	#		prompt = promptList[i]
	#		if i > 0: prompt2 = promptList[i - 1]
	#		else: prompt2 = prompt
	#		
	#		if prompt != prompt2 and modelchange[id(prompt)] != modelchange[id(prompt2)]:
	#			if len(promptGroup) > 0:
	#				promptGroups.append(promptGroup)
	#			promptGroups.append(prompt)
	#			promptGroup = []
	#		elif i % prompt.batch_size == 0:
	#			if promptGroup:
	#				promptGroups.append(promptGroup)
	#			promptGroup.append(prompt)
	#		else:
	#			promptGroup.append(prompt)
	#	if promptGroup:
	#		promptGroups.append(promptGroup)
#
	#	dataLog(f"All Groups made. \n There are {len(promptGroups)} groups after grouping. making batches now", True, 1)
#
	#	mergedPrompts = []
	#	
#	#	for it, prom in enumerate(promptGroups):
#	#		print("prompt group")
#	#		prom2 = self.batchPromptsValid(prom)
#	#		if isinstance(prom2, list):
#	#			mergedPrompts.extend(prom2)
#	#		elif prom2 is not None:
#	#			mergedPrompts.append(prom2)
#
	#	def bgroup(group):
	#		result = self.batchPromptsValid(group)
	#		if result is not None:
	#			with lock:
	#				mergedPrompts.append(result)
#
	#	for iterator, promptGroup in enumerate(promptGroups):
	#		threads = []
	#		thread = threading.Thread(target=bgroup, args=(promptGroup,))
	#		threads.append(thread)
	#		thread.start()
#
	#	for thread in threads:
	#		thread.join()
	#	return mergedPrompts
#
	#def batchPromptsValid(self, promptList):
	#	if isinstance(promptList, StableDiffusionProcessing) or isinstance(promptList, StableDiffusionProcessingTxt2Img) or isinstance(promptList, StableDiffusionProcessingImg2Img):
	#		promptList.batch_size = 1
	#		dataLog(f"single", True, 0)
	#		return promptList
	#	elif isinstance(promptList, int):
	#		dataLog(f"int", True, 0)
	#		return
	#	else:
	#		dataLog(f"group", True, 0)
	#		baseProm = promptList[0]
	#		for iterator, tempPrompt in enumerate(promptList):
	#			if not all(hasattr(tempPrompt2, attr) for tempPrompt2 in promptList for attr in dir(tempPrompt)):
	#				dataLog(f"prompt does not contain {str(attr)} can not merge", False, 0)
	#				return promptList
	#			for attr in dir(tempPrompt):
	#				if attr.startswith("__"): continue
	#				if callable(getattr(tempPrompt, attr)):continue
	#				if isinstance(getattr(tempPrompt, attr, None), types.BuiltinFunctionType) or isinstance(getattr(tempPrompt, attr, None), types.BuiltinMethodType): continue
	#				if attr in ['prompt', 'all_prompts', 'all_negative_prompts', 'negative_prompt', 'seed', 'subseed']: continue
	#				try:
	#					if getattr(tempPrompt, attr) == getattr(baseProm, attr): continue
	#					else: 
	#						if iterator == 1: 
	#							dataLog(f"Prompt contains incorrect {str(attr)} merge unavailable. values are: {str(getattr(tempPrompt, attr))}", False, 0)
	#						dataLog(f"prompt contains incorrect {str(attr)} merge unavailable. values are: {str(getattr(baseProm, attr))}", False, 0)
	#						return promptList
	#				except AttributeError:
	#					print(tempPrompt)
	#					dataLog(baseProm, False, 0)
	#					dataLog(f"failed to merged a group", True, 1)
	#					return promptList
	#		mergedPrompt = baseProm
	#		mergedPrompt.prompt = [p.prompt for p in promptList]
	#		mergedPrompt.negative_prompt = [p.negative_prompt for p in promptList]
	#		mergedPrompt.seed = [p.seed for p in promptList]
	#		mergedPrompt.subseed = [p.subseed for p in promptList]
	#		for prompt in promptList:
	#			setup2 = self.appliedSets.get(id(prompt), [])
	#			mergedFilePaths = [setup.filepath for setup in self.appliedSets[id(mergedPrompt)]]
	#			if any(setall.filepath in mergedFilePaths for setall in setup2): continue
	#			if self.appliedSets.get(id(prompt), []) in self.appliedSets[id(mergedPrompt)]: continue
	#			self.appliedSets[id(mergedPrompt)] += self.appliedSets.get(id(prompt), [])
	#		mergedPrompt.batch_size = len(promptList)
	#		dataLog(f"merged a group", True, 1)
	#		return mergedPrompt




def PostHandler(gridRunner: GridRunner, promptkey: StableDiffusionProcessing, appliedsets: dict) -> Processed:
	return
######################### Web Data Builders #########################

class WebDataBuilder():
	def buildJson(grid: GridFileHelper, publish_gen_metadata: bool, p, dryrun: bool):
		def get_axis(axis: str):
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
		show_descrip = grid.gridObj.get('show descriptions')
		result = {
			'title': grid.title,
			'description': grid.description,
			'ext': grid.format,
			'minWidth': grid.minWidth,
			'minHeight': grid.minHeight,
			'defaults': {
				'show_descriptions': True if show_descrip is None else show_descrip,
				'autoscale': grid.gridObj.get('autoscale') or False,
				'sticky': grid.gridObj.get('sticky') or False,
				'x': get_axis('x axis'),
				'y': get_axis('y axis'),
				'x2': get_axis('x super axis'),
				'y2': get_axis('y super axis')
			}
		}
		if not dryrun:
			result['will_run'] = True
		if publish_gen_metadata:
			result['metadata'] = None if webDataGetBaseParamData is None else webDataGetBaseParamData(p)
		axes = list()
		for axis in grid.axes:
			jAxis = {}
			jAxis['id'] = str(axis.id).lower()
			jAxis['title'] = axis.title
			jAxis['description'] = axis.description or ""
			values = list()
			for val in axis.values:
				jVal = {}
				jVal['key'] = str(val.key).lower()
				jVal['title'] = val.title
				jVal['description'] = val.description or ""
				jVal['show'] = val.show
				if publish_gen_metadata:
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
		return f'<input type="radio" class="btn-check" name="{name}" id="{str(id).lower()}" autocomplete="off" checked=""><label class="btn btn-outline-primary" for="{str(id).lower()}" title="{descrip}">{escapeHTML(label)}</label>\n'
	
	def axisBar(label, content):
		return f'<br><div class="btn-group" role="group" aria-label="Basic radio toggle button group">{label}:&nbsp;\n{content}</div>\n'

	def buildHtml(grid):
		with open(AssetDir + "/page.html", 'r', encoding="utf-8") as referenceHtml:
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
				content += f'<tr class="{trClass}">\n<td>\n<h4>{escapeHTML(axis.title)}</h4>\n'
				advancedSettings += f'\n<h4>{axis.title}</h4><div class="timer_box">Auto cycle every <input style="width:30em;" autocomplete="off" type="range" min="0" max="360" value="0" class="form-range timer_range" id="range_tablist_{axis.id}"><label class="form-check-label" for="range_tablist_{axis.id}" id="label_range_tablist_{axis.id}">0 seconds</label></div>\nShow value: '
				axisClass = "axis_table_cell"
				if len(axisDescrip.strip()) == 0:
					axisClass += " emptytab"
				content += f'<div class="{axisClass}">{axisDescrip}</div></td>\n<td><ul class="nav nav-tabs" role="tablist" id="tablist_{axis.id}">\n'
				primary = not primary
				isFirst = axis.default is None
				for val in axis.values:
					if axis.default is not None:
						isFirst = str(axis.default) == str(val.key)
					selected = "true" if isFirst else "false"
					active = " active" if isFirst else ""
					isFirst = False
					descrip = cleanForWeb(val.description or '')
					content += f'<li class="nav-item" role="presentation"><a class="nav-link{active}" data-bs-toggle="tab" href="#tab_{axis.id}__{val.key}" id="clicktab_{axis.id}__{val.key}" aria-selected="{selected}" role="tab" title="{escapeHTML(val.title)}: {descrip}">{escapeHTML(val.title)}</a></li>\n'
					advancedSettings += f'&nbsp;<input class="form-check-input" type="checkbox" autocomplete="off" id="showval_{axis.id}__{val.key}" checked="true" onchange="javascript:toggleShowVal(\'{axis.id}\', \'{val.key}\')"> <label class="form-check-label" for="showval_{axis.id}__{val.key}" title="Uncheck this to hide \'{escapeHTML(val.title)}\' from the page.">{escapeHTML(val.title)}</label>'
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
		html = html.replace("{TITLE}", grid.title).replace("{CLEAN_DESCRIPTION}", cleanForWeb(grid.description)).replace("{DESCRIPTION}", grid.description).replace("{CONTENT}", content).replace("{ADVANCED_SETTINGS}", advancedSettings).replace("{AUTHOR}", grid.author).replace("{EXTRA_FOOTER}", ExtraFooter).replace("{VERSION}", getVersion())
		return html

	def EmitWebData(path, publish_gen_metadata, p, yamlContent, dryrun: bool):
		global grid
		print("Building final web data...")
		os.makedirs(path, exist_ok=True)
		json = WebDataBuilder.buildJson(grid, publish_gen_metadata, p, dryrun)
		if not dryrun:
			with open(os.path.join(path, 'last.js'), 'w+', encoding="utf-8") as f:
				f.write("windows.lastUpdated = []")
		with open(path + "/data.js", 'w', encoding="utf-8") as f:
			f.write("rawData = " + json)
		if yamlContent is None:
			yamlContent = WebDataBuilder.buildYaml(grid)
		with open(path + "/config.yml", 'w', encoding="utf-8") as f:
			yaml.dump(yamlContent, f, sort_keys=False, default_flow_style=False, width=1000)
		for f in ["bootstrap.min.css", "jsgif.js", "bootstrap.bundle.min.js", "proc.js", "jquery.min.js", "styles.css", "placeholder.png"] + ExtraAssets:
			shutil.copyfile(AssetDir + "/" + f, path + "/" + f)
		html = WebDataBuilder.buildHtml(grid)
		with open(path + "/index.html", 'w', encoding="utf-8") as f:
			f.write(html)
		print(f"Web file is now at {path}/index.html")

######################### Main Runner Function #########################

def runGridGen(passThroughObj: StableDiffusionProcessing, inputFile: str, outputFolderBase: str, outputFolderName: str = None, doOverwrite: bool = False, generatePage: bool = True, publishGenMetadata: bool = True, dryRun: bool = False, manualPairs: list = None):
	global grid, logFile
	startTime = datetime.datetime.now()
	grid = GridFileHelper()
	yamlContent = None
	if manualPairs is None:
		fullInputPath = AssetDir + "/" + inputFile
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
	#folder = outputFolderBase + "/" + outputFolderName
	print("will save to: " + outputFolderName)
	if os.path.isabs(outputFolderName):
		folder = outputFolderName
	else:
		folder = os.path.join(outputFolderBase, outputFolderName)
	runner = GridRunner(doOverwrite, folder, passThroughObj)
	logFile = datalogNew(folder)
	cacheFile = os.path.join(folder, "cache")
	runner.preprocess()
	prerun = datetime.datetime.now()
	if generatePage:
		json = WebDataBuilder.EmitWebData(folder, publishGenMetadata, passThroughObj, yamlContent, dryRun)
	runnerStart = datetime.datetime.now()
	dataLog(f"Main runner started at {startTime}, pairing done at {pairTime}, preprocess at {prerun}, and processor started at {runnerStart} time for each: {quickTimeMath(startTime=startTime, endTime=pairTime)}, {quickTimeMath(startTime=startTime, endTime=prerun)}, {quickTimeMath(startTime=startTime, endTime=runnerStart)}", True, 1)
	result = runner.run(dryRun)
	if dryRun:
		print("Infinite Grid dry run succeeded without error")
	else:
		json = str(json).replace('"will_run": true, ', '')
		with open(os.path.join(folder, "data.js"), 'w', encoding="utf-8") as f:
			f.write("rawData = " + json)
		os.remove(os.path.join(folder, "last.js"))
	return result