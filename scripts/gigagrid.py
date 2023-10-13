
#imports
from modules import scripts
import os, concurrent.futures, gradio as gr, glob

#globals
ImagesCache: list[str]
AssetDir: str
imageTypes: list[str] = [".jpg", ".jpeg", ".png", ".webp"]

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
		else: return True
		
	
	def ui(self, is_img2img) -> None:
		listImageFiles()
		tryInit()
		with gr.Row():
			gridFile = gr.Dropdown(value="", label="Select grid File", choices=getNameList())
			self.refresh()
			def refresh() -> list[str]:
				newchoices = getNameList()
				gridFile.options = newchoices
				return newchoices
		gr.Interface([gridFile], refresh, live=True).launch()

		outputFilePath = gr.Textbox(value="", label="Output Folder")
		