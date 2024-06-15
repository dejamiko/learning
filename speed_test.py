import time

t = []

for j in range(5):
    l = []

    pTime = time.time()

    for i in range(10000000):
        l.append(i)

    print('test took: ' + str(time.time() - pTime) + ' units time')
    t.append(time.time() - pTime)

s = 0
for i in t:
    s = s + i

print('average time = ' + str(s / len(t)))
