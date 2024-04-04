import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from time import time, gmtime

#
# x			Unix-время, секунды
#

# Нормированная вероятность посетителя в определённый час дня
# На самом деле может быть суммой нескольких случайных величин:
# - В какое время идут обедать
# - В какое время уходят с работы (и идут есть)
def hour_of_day_visitor_distribution():
	x = np.random.normal(17, 3, [1000])
	h = np.bincount(np.int64(x % 24))
	n = h / h.sum()

	def debug():
		plt.hist(x % 24, bins=24)
		plt.show()

		plt.plot(n)
		plt.plot(np.cumsum(n))
		plt.show()

	return n

# Разметка дней недели на массиве временных меток
def weekdays(x):
	return np.array([gmtime(x).tm_wday for x in x])

# Разметка дней месяца на массиве временных меток
def monthdays(x):
	return np.array([gmtime(x).tm_mday for x in x])

# Разметка месяцев на массиве временным меток
def months(x):
	return np.array([gmtime(x).tm_mon for x in x])

# Посетителей в год
def visitors(days=365):
	daily = np.random.normal(200, 5, [days])
	tz = 3600*8 # GMT+8

	# Начало данных в полночь
	x = np.arange(days*24)*3600 + (time() // 3600 // 24 * 24 * 3600)
	y = daily[:, None] * np.vstack([hour_of_day_visitor_distribution() for i in range(days)])
	y = y.flatten()

	# Предположение: в выходные в два раза меньше посетителей
	# Естественно, это зависит от расположения
	weekend_factor = np.zeros_like(x) + (weekdays(x) == 5) + (weekdays(x) == 6)
	weekend_factor = (np.zeros_like(x) + 1) - (weekend_factor * 0.5)
	y = y * weekend_factor

	def debug():
		plt.figure(figsize=[16, 9], dpi=150)

		plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
		plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
		plt.setp(plt.gca().get_xticklabels(), rotation=30, ha="right")

		plt.step(x/3600/24, y)
		plt.show()

	return x, y

# Каждое первое число месяца платим аренду за помещение
def rent(x):
	return np.convolve(monthdays(x) == 1, [1, -1], mode="same") > 0

# Попробуем смоделировать вместимость
x, y = visitors(365)

occupancy = y.clip(0, 20)
overcapacity = y - occupancy

# Просто предположим, что всегда покупают на 200 +- 100 руб
avg_checks = np.abs( np.random.normal(200, 100, y.shape) )
income = occupancy * avg_checks

profit = np.cumsum(income - rent(x)*200000)

plt.figure(figsize=[16, 9], dpi=150)

# Форматирование даты и времени
# Автоматический выбор интервала подписей
# ISO8601 время: "%Y-%m-%d %H:%M:%S"
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
plt.setp(plt.gca().get_xticklabels(), rotation=30, ha="right")

plt.step(x/3600/24, occupancy)
#plt.step(x/3600/24, profit)
plt.tight_layout()
plt.show()
