from bigcrittercolor.imgdownload import getiNatGenusNames
import itertools

# more than 10,000 Beetle genera so have to split them up LMAO
l1 = getiNatGenusNames.getiNatGenusNames("Adephaga")
l2 = getiNatGenusNames.getiNatGenusNames("Archostemata")
l3 = getiNatGenusNames.getiNatGenusNames("Myxophaga")
l4 = getiNatGenusNames.getiNatGenusNames("Bostrichiformia")
l5 = getiNatGenusNames.getiNatGenusNames("Elateriformia")
l6 = getiNatGenusNames.getiNatGenusNames("Scarabaeiformia")
l7 = getiNatGenusNames.getiNatGenusNames("Staphyliniformia")
l8 = getiNatGenusNames.getiNatGenusNames("Cerambycoidea")
l9 = getiNatGenusNames.getiNatGenusNames("Chrysomeloidea")
#l10 = getiNatGenusNames.getiNatGenusNames("Cleroidea")
#l11 = getiNatGenusNames.getiNatGenusNames("Coccinelloidea")
#l12 = getiNatGenusNames.getiNatGenusNames("Cucujoidea")
#l13 = getiNatGenusNames.getiNatGenusNames("Curculionoidea")
#l14 = getiNatGenusNames.getiNatGenusNames("Lymexyloidea")
#l15 = getiNatGenusNames.getiNatGenusNames("Tenebrionoidea")

combined_list = list(itertools.chain(l1,l2,l3,l4,l5,l6,l7,l8,l9))#,l10))#,l11,l12,l13,l14,l15))

print(combined_list)