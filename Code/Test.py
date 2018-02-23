import timing, time

# Used to test small code snippets, just to test other stuff. 

j = 0
for i in range(0, 10):
    j =+ i
    time.sleep(0.1)
    timing.log('Important part here!')

print(j)