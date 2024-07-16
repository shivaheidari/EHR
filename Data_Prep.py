from sqlalchemy import create_engine
from sqlalchemy import URL
from sqlalchemy import text
#hostname: 127.0.0.1
#port = 3306
#username: root
#password: root
engine = create_engine("mysql+pymysql://root:root@localhost:3306/MIMIC")

with engine.connect() as conn:
    query = text ("SELECT count(*) FROM MIMIC.ADMISSIONS;")
    result = conn.execute(query)
    count = result.fetchone()[0]
    print("number of admissions: ", count)
    # for row in result:
    #     print(row)

