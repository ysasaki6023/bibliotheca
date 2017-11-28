import bibliotheca.base

if __name__=="__main__":
    bib = bibliotheca.base(userID="myName",userPass="myPass")
    func1 = bib.get(objectName="objName1",objectID="XXXXX-XXXXX-XXXXX")
    func2 = bib.get(objectName="objName2",objectID="XXXXX-XXXXX-XXXXX")
