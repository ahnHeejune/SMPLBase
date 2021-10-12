###################################################################
# Chumpy automatic  
# (c) 2020  heejune@seoultech.ac.kr
###################################################################

import chumpy 

        
a = chumpy.ones(5)
b = chumpy.ones(5)
c = a + b
print(c.r)
print(type(c))

# the automatic update 
a[:] = 5
print(a.r)
print(c.r)
