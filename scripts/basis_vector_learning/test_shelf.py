import shelve

shelf = shelve.open("/tmp/shelve.out")

for key in shelf:
    globals()[key]=shelf[key]
    print(shelf[key])

shelf.close()