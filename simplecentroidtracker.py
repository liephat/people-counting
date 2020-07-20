from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

"""
This class is used to track detected objects during a sequence of frames.

It is based on an algorithm that was created by Adrian Rosebrock (pyimagesearch.com) and it was slightly modified to
meet the requirements of counting people in a video.

"""

class SimpleCentroidTracker:
	def __init__(self, max_disappeared=50, max_distance=50):

		self.next_object_id = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		self.object_count = 0

		self.max_disappeared = max_disappeared

		self.maxDistance = max_distance

	def get_count(self):
		return self.object_count

	def update(self, rects):
		if len(rects) == 0:

			for object_id in list(self.disappeared.keys()):
				self.disappeared[object_id] += 1

				if self.disappeared[object_id] > self.max_disappeared:
					self.deregister(object_id)

			return self.objects

		input_centroids = np.zeros((len(rects), 2), dtype="int")

		for (i, (start_x, start_y, end_x, end_y)) in enumerate(rects):
			c_x = int((start_x + end_x) / 2.0)
			c_y = int((start_y + end_y) / 2.0)
			input_centroids[i] = (c_x, c_y)

		if len(self.objects) == 0:
			for i in range(0, len(input_centroids)):
				self.register(input_centroids[i])

		else:

			object_ids = list(self.objects.keys())
			object_centroids = list(self.objects.values())

			D = dist.cdist(np.array(object_centroids), input_centroids)

			rows = D.min(axis=1).argsort()

			cols = D.argmin(axis=1)[rows]

			used_rows = set()
			used_cols = set()

			for (row, col) in zip(rows, cols):

				if row in used_rows or col in used_cols:
					continue

				if D[row, col] > self.maxDistance:
					continue

				object_id = object_ids[row]
				self.objects[object_id] = input_centroids[col]
				self.disappeared[object_id] = 0

				used_rows.add(row)
				used_cols.add(col)

			unused_rows = set(range(0, D.shape[0])).difference(used_rows)
			unused_cols = set(range(0, D.shape[1])).difference(used_cols)

			if D.shape[0] >= D.shape[1]:

				for row in unused_rows:
					object_id = object_ids[row]
					self.disappeared[object_id] += 1

					if self.disappeared[object_id] > self.max_disappeared:
						self.deregister(object_id)

			else:
				for col in unused_cols:
					self.register(input_centroids[col])

		return self.objects

	def register(self, centroid):
		self.objects[self.next_object_id] = centroid
		self.disappeared[self.next_object_id] = 0
		self.next_object_id += 1
		self.object_count += 1

	def deregister(self, objectID):
		del self.objects[objectID]
		del self.disappeared[objectID]
