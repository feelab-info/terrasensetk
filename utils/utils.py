from os import path
import datetime

def get_lucas_copernicus_path():
    lucas_path = path.dirname(__file__)
    return path.join(lucas_path,"lucas_ds.pkl")


def get_time_interval(middle_date, number_of_days):
    """
        Gets the time interval surrounding the middle date separated by slashes
    Args:
        middle_date: A string containing the date which will be included in the timerange
        number_of_days: The number of days counting from the `middle_date` that will correspond to the min and max date
        
    Returns:
        A list with the ´number_of_days´ before and after of the ´middle_date´
        
    Example:
        >>> get_time_interval("15/09/1998", 3)
        >>> ['1998-09-12', '1998-09-18']
        
    """
    point_date = datetime.datetime.strptime(middle_date, '%d/%m/%y')
    days_before = point_date - datetime.timedelta(days=number_of_days)
    days_after = point_date + datetime.timedelta(days=number_of_days)
    return [days_before.strftime('%Y-%m-%d'), days_after.strftime('%Y-%m-%d')]


def get_indexes_from_bands():
    link = r'https://www.indexdatabase.de/db/is.php?sensor_id=96'
    start = 450
    step = 4.32
    bands_dic = [];
    for i in range(0,116):
        bands_dic.append({"BAND_ID":i, "range":(round(start+(step*i),2),round(start+step*(i+1),2))})
        #print(f'Band {i}: {start+(step*i):.2f} - {start+(step*(i+1)):.2f}')

    print(bands_dic)

from lxml import etree
def parse_table_to_dict(html_table):
    table = etree.HTML(html_table).find("body/table")
    rows = iter(table)
    headers = [col.text for col in next(rows)]
    for row in rows:
        values = [col.text for col in row]
        print (zip(headers, values))
        
from shapely.geometry import Point
from math import sqrt

def to_square(polygon):
    pol = polygon.buffer(0.0005)
    
    minx, miny, maxx, maxy = pol.bounds
    
    # get the centroid
    centroid = [(maxx+minx)/2, (maxy+miny)/2]
    # get the diagonal
    diagonal = sqrt((maxx-minx)**2+(maxy-miny)**2)
    
    return Point(centroid).buffer(diagonal/2, cap_style=3)
#target third td from the table with class matrix
#get_indexes_from_bands()

#from html.parser import HTMLParser
#import urllib.request as urllib2
#from bs4 import BeautifulSoup as bs

#link = r'https://www.indexdatabase.de/db/is.php?sensor_id=96'

#html_page = urllib2.urlopen(link)
#soup = bs(html_page)
#tables = soup.findAll("table")
#print(tables[0])
#procurar dentro dos mi o valor e comparar com as merdinhas que tenho

#for table in tables:
     #if table.findParent("table") is None:
         #print(str(table))
#print(str(html_page.read())[:200])
#parse_table_to_dict(str(html_page.read()))

