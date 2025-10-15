import matplotlib.pyplot as plt

tasks = ["DreamTrip", "MakeMyTrip"]
completion_time = [12, 18]  # in minutes
satisfaction = [4.4, 3.1]   # 1–5 scale

# Task Completion Time
plt.bar(tasks, completion_time, color=['skyblue', 'orange'])
plt.title("Task Completion Time Comparison (minutes)")
plt.show()

# Satisfaction Ratings
plt.bar(tasks, satisfaction, color=['green', 'red'])
plt.title("Satisfaction Ratings Comparison (1–5)")
plt.show()
