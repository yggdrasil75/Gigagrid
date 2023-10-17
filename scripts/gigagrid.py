
#imports
import os, concurrent.futures, gradio as gr, glob, hashlib, threading, re, logging
from modules import sd_models as sdModels, scripts
from modules.shared import opts
from colorama import init as CInit, Fore, Style

#globals
ImagesCache: list[str] = None
AssetDir: str = os.path.dirname(__file__) + "/assets"
imageTypes: list[str] = [".jpg", ".jpeg", ".png", ".webp"]
validModes: list = {}
checkPointList: list[str] = {}
cleanList: dict = {}
logFile: str = "/Log.txt"
printLevel: int = 1

#generic functions
""" def listImageFiles():
	global ImagesCache, AssetDir, imageTypes
	if ImagesCache is not None:
		return ImagesCache
	ImagesCache = []
	imageDir = os.path.join(AssetDir, "images")
	for root, _, files in os.walk(imageDir):
		for file in files:
			_, ext = os.path.splitext(file)
			if ext.lower() in imageTypes:
				fn = os.path.relpath(os.path.join(root, file),imageDir)
				ImagesCache.append(fn)
	return ImagesCache """

def listImageFiles() -> list[str]:
	global ImagesCache, AssetDir, imageTypes
	if ImagesCache is not None:
		return ImagesCache
	
	imageDir = os.path.join(AssetDir, "images")
	with concurrent.futures.ThreadPoolExecutor() as executor:
		imageFiles = []

		def listImageFilesWorker(root):
			ImageFiles = []
			for file in os.listdir(root):
				_, ext = os.path.splitext(file)
				if ext.lower() in imageTypes:
					ImageFiles.append(os.path.relpath(os.path.join(root, file), imageDir))
			return ImageFiles


		for root, _, _ in os.walk(imageDir):
			future = executor.submit(listImageFilesWorker, root, imageTypes)
			imageFiles.append(future)
		
		ImagesCache = [fn for future in concurrent.futures.as_completed(imageFiles) for fn in future.result()]

	return ImagesCache

def getNameList() -> list[str]:
	global AssetDir
	fileList = glob.glob(AssetDir + "/*.yml")
	fileList.extend(glob.glob(opts.outdir_txt2img_grids + "/*.yml"))
	fileList.extend(glob.glob(opts.outdir_grids + "/*.yml"))
	fileList.extend(glob.glob(opts.outdir_img2img_grids + "/*.yml"))
	justFileNames = sorted(list(map(lambda f: os.path.relpath(f, AssetDir), fileList)))
	return justFileNames

#classes
class Script(scripts.Script):
	basedir = scripts.basedir()
	def title(self) -> str:
		if __file__.endswith('.pyc'):
			return "Gigagrid (compiled)"
		else:
			return 'Gigagrid - do not use yet. use "Generate Infinite-Axis Grid v2"'
	
	def show(self, is_img2img) -> bool:
		if is_img2img: return False
		else: return False
		
		
	def refresh(self, gridFile) -> list[str]:
		newchoices = getNameList()
		gridFile.options = newchoices
		return newchoices
	
	def ui(self, is_img2img) -> None:
		return
		listImageFiles()
		tryInit()
		with gr.Row():
			gridFile = gr.Dropdown(value="", label="Select grid File", choices=getNameList())
			self.refresh(gridFile)
			
		outputFilePath = gr.Textbox(value="", label="Output Folder")
		gr.Interface(inputs=[gridFile], outputs=[outputFilePath], fn=self.refresh, live=True).launch()

		
#mode inits
def tryInit():
	global checkPointList
	checkPointList = list(map(lambda m: m.title, sdModels.checkpoints_list.values()))
	registerMode("model", gridSettingMode(type=str,apply=applyModel, clean=cleanModel, validList=lambda: list(map(lambda m: m.title, sdModels.checkpoint_list.values()))))

def registerMode(Name:str, Mode):
	Mode.Name = cleanMode(Name)
	validModes[Mode.Name] = Mode

class gridSettingMode:
	def __init__(self, type: type, apply: callable, min: float = None, max: float = None, validList: callable = None, clean:callable = None):
		self.type = type
		self.apply = apply
		self.min = min
		self.max = max
		self.clean = clean
		self.validList = validList

def cleanMode(name:str) -> str:
	return name.lower().replace('[', '').replace(']', '').replace(' ', '').replace('\\', '/').replace('//', '/').strip()

def applyModel(p, model):
	return getBestInList(model, checkPointList)

def cleanModel(p, model):
	actualModel = getBestInList(model, checkPointList)
	if actualModel is None:
		raise RuntimeError(f"Invalid parameter '{p}' as '{model}': model name unrecognized - valid {list(checkPointList)}")
	return model

#utilities
def getQuickList(list):
	global quickListCache
    # Calculate a hash for the input list.
	listHash = hashlib.sha1(str(list).encode()).hexdigest()

    # Check if the list is already cached, and if so, return it.
	if listHash in quickListCache:
		logging.info(f"[getQuickList] using cache")
		return quickListCache[listHash]
	else: 
		logging.info(f"[getQuickList] not using cache")

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
	logging.info(f"[getBestInList] name = {name} clean = {clean}")

    # Get the quick list for the input list.
	quickList = getQuickList(list)

	if clean in quickList:
		return quickList[clean][0]
	else:
		return None
	
def cleanName(name: str) -> str:
	global cleanList
	if name in cleanList: 
		return cleanList[name]
	else:
		# Use regular expressions to remove everything between square brackets
		cleanedName = re.sub(r'\[.*?\]', '', name)
		# Convert to lowercase, replace spaces with '', and normalize slashes
		cleanedName = cleanedName.lower().replace(' ', '').replace('\\', '/').replace('//', '/').strip()
		cleanList[name] = cleanedName
		return cleanedName
	
def setupLogging():
	return