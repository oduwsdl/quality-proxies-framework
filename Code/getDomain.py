from tldextract import extract #pip install tldextract

def getDomain(url, includeSubdomain=False, excludeWWW=True):

    url = url.strip()
    if( len(url) == 0 ):
        return ''

    if( url.find('http') == -1  ):
        url = 'http://' + url

    domain = ''
    
    try:
        ext = extract(url)
        
        domain = ext.domain.strip()
        subdomain = ext.subdomain.strip()
        suffix = ext.suffix.strip()

        if( len(suffix) != 0 ):
            suffix = '.' + suffix 

        if( len(domain) != 0 ):
            domain = domain + suffix
        
        if( excludeWWW ):
            if( subdomain.find('www') == 0 ):
                if( len(subdomain) > 3 ):
                    subdomain = subdomain[4:]
                else:
                    subdomain = subdomain[3:]


        if( len(subdomain) != 0 ):
            subdomain = subdomain + '.'

        if( includeSubdomain ):
            domain = subdomain + domain
    except:
        genericErrorInfo()
        return ''

    return domain

print( getDomain('http://example.com') )