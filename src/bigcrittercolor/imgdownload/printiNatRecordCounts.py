from bigcrittercolor.helpers import _bprint, _inat_dl_helpers

def printiNatRecordCounts(taxon_id=47792,research_grade=True,lat_lon_box=((24.396308, -124.848974), (49.384358, -66.885444))):
    base_params = {'taxon_id': taxon_id}
    if research_grade:
        base_params['quality_grade'] = 'research'
    if lat_lon_box is not None:
        base_params['swlat'] = lat_lon_box[0][0]
        base_params['swlng'] = lat_lon_box[0][1]
        base_params['nelat'] = lat_lon_box[1][0]
        base_params['nelng'] = lat_lon_box[1][1]
    count = _inat_dl_helpers.getRecCnt(base_params)
    print(str(count) + " records")

#printiNatRecordCounts(research_grade=True)
#printiNatRecordCounts(research_grade=False)

