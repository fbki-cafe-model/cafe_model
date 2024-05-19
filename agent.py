from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_percentage_error
from matplotlib.dates import AutoDateLocator, DateFormatter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("RestaurantDataVets_All_2to5.csv", na_values="?", parse_dates=[2])
df = df.dropna()

def filter():
	q_low = df["2to5"].quantile(0.01)
	q_hi  = df["2to5"].quantile(0.99)

	return df[(df["2to5"] < q_hi) & (df["2to5"] > q_low)]

df = filter()

"""
>>> df.dtypes
Index                       int64
Group                       int64
DMY                datetime64[ns]
MissingPrevDays             int64
Year                        int64
Day                         int64
January                     int64
February                    int64
March                       int64
April                       int64
May                         int64
June                        int64
July                        int64
August                      int64
September                   int64
October                     int64
November                    int64
December                    int64
Sunday                      int64
Monday                      int64
Tuesday                     int64
Wednesday                   int64
Thursday                    int64
Friday                      int64
Saturday                    int64
Holiday                     int64
Carnival                    int64
LentFasting                 int64
Ramadan                     int64
ChristmasSeason             int64
DailyAvg                  float64
WeeklyAvg                 float64
MinSales                  float64
MaxSales                  float64
DailyBusyness             float64
WeeklyBusyness            float64
2to5                      float64
dtype: object
"""



def overview():
	fig, axes = plt.subplots()

	axes.step(df.DMY, df["2to5"])

	axes.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d (%a)"))
	fig.autofmt_xdate()

	plt.show()

# V2 with weekdays numbered in increasing order
# Notable:
# - Must not be a plain cumsum, only provides a linear increase if added as such
# - A better fit with sqrt
def linear_v2():
	time = np.float64(df.DMY)[:, None]
	day_idx = time / 1000 / 1000/ 1000 / 60 / 60 / 24

	y = df["2to5"]
	x = np.hstack([
		#time / time.max(),
		#np.float64(df.Monday)[:, None],
		#np.float64(df.Tuesday)[:, None],
		#np.float64(df.Wednesday)[:, None],
		#np.float64(df.Thursday)[:, None],
		#np.float64(df.Friday)[:, None],
		#np.float64(df.Saturday)[:, None],
		#np.float64(df.Sunday)[:, None],
		np.float64(df.Monday.cumsum() * df.Monday)[:, None] ** 0.3,
		np.float64(df.Tuesday.cumsum() * df.Tuesday)[:, None] ** 0.3,
		np.float64(df.Wednesday.cumsum() * df.Wednesday)[:, None] ** 0.3,
		np.float64(df.Thursday.cumsum() * df.Thursday)[:, None] ** 0.3,
		np.float64(df.Friday.cumsum() * df.Friday)[:, None] ** 0.3,
		np.float64(df.Saturday.cumsum() * df.Saturday)[:, None] ** 0.3,
		np.float64(df.Sunday.cumsum() * df.Sunday)[:, None] ** 0.3,
	])

	train_x, test_x = x[:800], x[800:]
	train_y, test_y = y[:800], y[800:]

	model = Lasso()
	model.fit(train_x, train_y)
	print("Score:", model.score(test_x, test_y) )

	preds = model.predict(x)

	print("MAE:", mean_absolute_error(y, preds))

	# Visuals
	fig, axes = plt.subplots()

	# Ground truth
	axes.step(df.DMY, df["2to5"])
	axes.step(df.DMY, preds)

	axes.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d (%a)"))
	fig.autofmt_xdate()

	plt.show()

class Agent():
	def __init__(self):
		self.planned_capacity = [400]*7
		self.leftover_queue = [0]*7

		self.history_leftovers = []
		self.history_expenses = []
		self.history_income = []
		self.history_expired = []
		self.history_demand = []
		self.history_plan = []
		self.history_met = []
		self.history_ts = []

		self.fig = None
		self.axes = None

	def visual_init(self):
		self.fig, self.axes = plt.subplots(2, 2)
		self.fig.tight_layout()

	def advance(self, ts, demand):
		expired = self.leftover_queue.pop(0)
		plan = self.planned_capacity.pop(0)
		surplus = max(plan - demand, 0)
		deficit = max(demand - plan, 0)

		self.history_expired.append(expired)
		self.history_demand.append(demand)
		self.history_plan.append(plan)
		self.history_ts.append(ts)

		leftovers = 0

		if surplus:
			met = demand
		elif deficit:
			leftovers = self.tap_surplus(deficit)
			met = demand - deficit + leftovers
		else:
			print("Exact prediction!")
			met = demand

		self.leftover_queue.append(surplus)
		self.history_leftovers.append(leftovers)
		self.history_met.append(met)

		self.history_expenses.append(plan * .25)
		self.history_income.append(met * .27)

		if not self.planned_capacity:
			self.plan()

	def plan(self):
		#self.planned_capacity = [400]*7
		#self.planned_capacity = [ int( sum(self.history_demand) / len(self.history_demand) ) ]
		#self.planned_capacity = self.history_demand[-1:]
		self.planned_capacity = self.history_demand[-7:]

	def tap_surplus(self, shortage):
		sourced = 0

		for i, _ in enumerate(self.leftover_queue):
			take = min(self.leftover_queue[i], shortage)
			self.leftover_queue[i] -= take
			shortage -= take
			sourced += take

		return sourced

	def visualize(self):
		if not self.fig:
			self.visual_init()

		demand = self.axes[0][0]
		losses = self.axes[0][1]
		profit = self.axes[1][0]

		demand.clear()
		demand.set_xlim(self.history_ts[0], self.history_ts[-1].ceil("30D"))
		demand.step(self.history_ts, self.history_demand, label="Спрос")
		demand.step(self.history_ts, self.history_met, label="Удовлетворено")
		demand.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d (%a)"))
		demand.legend()

		losses.clear()
		losses.set_xlim(self.history_ts[0], self.history_ts[-1].ceil("30D"))
		losses.step(self.history_ts, self.history_expired, c="red", label="Истёк срок годности")
		losses.step(self.history_ts, self.history_leftovers, c="orange", label="Продано накопленных излишков")
		losses.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d (%a)"))
		losses.legend()

		cum_income = np.cumsum(self.history_income)
		cum_expenses = np.cumsum(self.history_expenses)

		profit.clear()
		profit.set_xlim(self.history_ts[0], self.history_ts[-1].ceil("30D"))
		profit.step(self.history_ts, cum_income - cum_expenses, c="lime", label="Баланс")
		profit.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d (%a)"))
		profit.legend()

		self.fig.autofmt_xdate()
		plt.pause(.01)

	def score(self):
		print("MAPE:", mean_absolute_percentage_error(self.history_demand, self.history_plan))

def test_surplus_usage():
	agent = Agent()
	agent.planned_capacity = [400]*7

	agent.advance(1, 200) # 200
	agent.advance(2, 200) # 400
	agent.advance(3, 200) # 600
	agent.advance(4, 200) # 800
	agent.advance(5, 200) # 1000
	agent.advance(6, 200) # 1200
	agent.advance(7, 200) # 1400, plan 200

	assert agent.leftover_queue == [200]*7

	agent.planned_capacity = [200]
	agent.advance(8, 1400) # 200 burns, 200 planned, 1200 surplus

	assert agent.leftover_queue == [0]*7
	assert agent.history_met == [200]*7 + [1400]

def selftest():
	test_surplus_usage()
	print("Self-testing OK")

def play_game():
	agent = Agent()

	time = np.float64(df.DMY)
	demand = df["2to5"]

	for ts, d in zip(df.DMY, demand):
		agent.advance(ts, d)
		#agent.visualize()

	agent.visualize()
	plt.show()
	agent.score()

selftest()
play_game()
