from collections import OrderedDict
import os
import threading
import pickle
from functools import wraps
import hashlib

class QuickCache:
	def __init__(self, maxSize: int, localCache: str | None, periodicTime=600):
		global cacheFile
		self.cache: OrderedDict = OrderedDict()  # Use an OrderedDict
		self.localCache = localCache
		self.cacheFile: str = self.localCache or cacheFile
		self.cachePath: str = os.path.dirname(os.path.realpath(self.cacheFile))
		if not os.path.exists(self.cachePath):
			os.makedirs(self.cachePath)
		self.maxSize: int = maxSize
		self.lock = threading.Lock()
		self.periodicTime: int = periodicTime
		self.PeriodicCacheWrite()

		try:
			self.LoadCache()
		except (FileNotFoundError, EOFError) as e:
			#DataLog(f"Failed to load cache: {e}", True, 0)
			print(f"failed to load cache: {e}")

	def __call__(self, func) -> callable:
		return self.Decorator(func)

	def LoadCache(self) -> dict:
		with self.lock:
			if os.path.exists(self.cacheFile):
				try:
					with open(self.cacheFile, 'rb') as file:
						tempCache = pickle.load(file)
						self.cache.update(tempCache)
				except (FileNotFoundError, EOFError) as e:
					#datalog(f"Failed to load cache: {e}", True, 0)
					print(f"failed to load cache: {e}")
		return self.cache

	def SaveCache(self) -> None:
		with self.lock:
			with open(self.cacheFile, 'wb') as file:
				pickle.dump(self.cache, file,fix_imports=True)

	def PeriodicCacheWrite(self):
		if self.periodicTime != 0:
			threading.Timer(self.periodicTime, self.PeriodicCacheWrite).start()
		self.SaveCache()

	def Decorator(self, func):

		@wraps(wrapped=func)
		def Wrapper(*args, **kwargs):
			keyArgs = []
			for arg in args:
				try:
					keyArgs.append(hash(str(arg).encode()).hexdigest())
				except:
					keyArgs.append(hashlib.blake2s(str(arg).encode()).hexdigest())


			key = (tuple(keyArgs), frozenset(kwargs.items()))

			# Check if the key is in the cache
			if key in self.cache:
				# Move the accessed item to the end of the ordered dictionary
				value = self.cache.pop(key)
				self.cache[key] = value
				return value

			result = func(*args, **kwargs)
			with self.lock:
				if len(self.cache) >= self.maxSize:
					# Remove the oldest entry (the first item in the ordered dictionary)
					self.cache.popitem(last=False)

				self.cache[key] = result

			return result

		return Wrapper