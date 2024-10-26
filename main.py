from letterboxdpy import user
from pre import findCompatibility, enhanced_compatibility

uname1 = input("enter name 1 : ")
uname2 = input("enter name 2 : ")

uobj1 = user.User(uname1)
uobj2 = user.User(uname2)

u1diary = user.user_diary(uobj1)
u2diary = user.user_diary(uobj2)

print(enhanced_compatibility(u1diary , u2diary))