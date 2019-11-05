import matplotlib.pyplot as plt
import numpy as np


def f(var):
    return var ** 2


x = np.arange(0, 10, 0.1)
y = f(x)

######################################
# Fill between a curve and the x axis
######################################
plt.plot(x, y, 'k--')

plt.fill_between(x, y, color='#539ecd')

plt.grid()

plt.title('How to fill an area in matplotlib ?', fontsize=10)

# plt.savefig('how_to_fill_area_matplotlib_01.png', bbox_inches='tight')
plt.show()
# plt.close()

######################################
# Fill the opposite area
######################################
plt.plot(x, y, 'k--')

plt.fill_between(x, y, np.max(y), color='#539ecd')

plt.grid()

plt.title('How to fill an area in matplotlib ?', fontsize=10)

# plt.savefig('how_to_fill_area_matplotlib_02.png', bbox_inches='tight')
plt.show()


# plt.close()

######################################
# Fill between two curves
######################################


def f1(var):
    return 1.0 / np.exp(var)


def f2(var):
    return np.log(var)


x = np.arange(0.01, 10, 0.1)

y1 = f1(x)
y2 = f2(x)

plt.plot(x, y1, 'k--')
plt.plot(x, y2, 'k--')

plt.fill_between(x, y1, y2, color='#539ecd')

plt.grid()

plt.xlim(0, 10)
plt.ylim(-1, 2.5)

plt.title('How to fill an area in matplotlib ?', fontsize=10)

# plt.savefig('how_to_fill_area_matplotlib_03.png', bbox_inches='tight')
plt.show()


# plt.close()


#############################################
# Fill between two curves using a condition
#############################################


def f1(var):
    return 1.0 / np.exp(var)


def f2(var):
    return np.log(var)


x = np.arange(0.01, 10, 0.1)

y1 = f1(x)
y2 = f2(x)

plt.plot(x, y1, 'k--')
plt.plot(x, y2, 'k--')

plt.fill_between(x, y1, y2, where=y1 < y2, color='#539ecd')

plt.grid()

plt.xlim(0, 10)
plt.ylim(-1, 2.5)
plt.title('How to fill an area in matplotlib ?', fontsize=10)

# plt.savefig('how_to_fill_area_matplotlib_04.png', bbox_inches='tight')
plt.show()
# plt.close()


#############################################
# Fill between two curves using a condition 2
#############################################
x = np.arange(0.01, 10, 0.1)

y1 = x
y2 = - x + np.max(y1)

y = np.minimum(y1, y2)

plt.plot(x, y1, 'k--')
plt.plot(x, y2, 'k--')

plt.fill_between(x, y, color='#539ecd')

plt.grid()

plt.title('How to fill an area in matplotlib ?', fontsize=10)

# plt.savefig('how_to_fill_area_matplotlib_05.png', bbox_inches='tight')
plt.show()
# plt.close()
