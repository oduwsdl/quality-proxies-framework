#pip install googlemaps
import googlemaps

def normalizeLoc(loc, googlemapsKey):

    loc = loc.strip()
    googlemapsKey = googlemapsKey.strip()
    normalizedLocation = {'location_name': loc, 'coordinates': {}}

    if( loc == '' or googlemapsKey == '' ):
        return normalizedLocation

    places = []
    try:
        gmaps = googlemaps.Client(key=googlemapsKey)
        places = gmaps.places(loc)
    except:
        print('Catch error here')

    if( len(places) == 0 ):
        return normalizedLocation

    if( 'results' in places ):
        if( len(places['results']) == 1 ):

            try:
                normalizedLocation['location_name'] = places['results'][0]['formatted_address']
                normalizedLocation['coordinates'] = places['results'][0]['geometry']['location']
            except:
                print('Catch error here')

    return normalizedLocation

'''
How to get your Google Maps Key:
1. Created account on Google Cloud Platform (https://console.cloud.google.com/) and Project
2. Activated Places API from Google Cloud Platform Dashboard
3. Enable billing (You're given free trial credit of $300 for 12 months, subsequently you may pay as you go if you choose to upgrade)
'''

yourGoogleMapsKey = ''
print( normalizeLoc("NYC", yourGoogleMapsKey) )
#{'location_name': 'New York, NY, USA', 'coordinates': {'lat': 40.7127753, 'lng': -74.0059728}}