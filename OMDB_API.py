import requests

def get_movies_details(title, api_key):
    url=f"http://www.omdbapi.com/?t={title}&plot=full&apikey={http://www.omdbapi.com/?i=tt3896198&apikey=150249eb}"
    res = requests.get(url).json()

    if res.get('response') == "True":
        result = res.get("plot," "N/A "), res.get("poster", "N/A")
        plot = result[0]
        poster = result[1]
        return plot , poster 
    
    return "N/A" , "N/A"