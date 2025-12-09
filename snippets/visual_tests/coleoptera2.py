from bigcrittercolor.imgdownload import getiNatGenusNames
import itertools

l10 = getiNatGenusNames.getiNatGenusNames("Cleroidea")
l11 = getiNatGenusNames.getiNatGenusNames("Coccinelloidea")
l12 = getiNatGenusNames.getiNatGenusNames("Cucujoidea")
l13 = getiNatGenusNames.getiNatGenusNames("Curculionoidea")
l14 = getiNatGenusNames.getiNatGenusNames("Lymexyloidea")
l15 = getiNatGenusNames.getiNatGenusNames("Tenebrionoidea")

combined_list = list(itertools.chain(l10,l11,l12,l13,l14,l15))

print(combined_list)