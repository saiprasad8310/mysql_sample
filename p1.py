import mysql.connector

mydb=mysql.connector.connect(
    host="localhost",
    user="root",
    password="saiprasad@8310",
    database="mydatabase2"
)

#rint(mydb)

#mycursor = mydb.cursor()

#ycursor.execute("CREATE DATABASE database1")

#ycursor.execute("SHOW DATABASES")

#mycursor.execute("CREATE TABLE customers (name VARCHAR(255), address VARCHAR(255))")    



"""mycursor = mydb.cursor()

sql = "INSERT INTO customers (name, address) VALUES (%s, %s)"
val = ("John", "Highway 21")
mycursor.execute(sql, val)

mydb.commit()
print(mycursor.rowcount, "record inserted.")"""

"""mycursor = mydb.cursor()

sql = "INSERT INTO customers (name, address) VALUES (%s, %s)"
val = [
  ('Peter', 'Lowstreet 4'),
  ('Amy', 'Apple st 652'),
  ('Hannah', 'Mountain 21'),
  ('Michael', 'Valley 345'),
  ('Sandy', 'Ocean blvd 2'),
  ('Betty', 'Green Grass 1'),
  ('Richard', 'Sky st 331'),
  ('Susan', 'One way 98'),
  ('Vicky', 'Yellow Garden 2'),
  ('Ben', 'Park Lane 38'),
  ('William', 'Central st 954'),
  ('Chuck', 'Main Road 989'),
  ('Viola', 'Sideway 1633')
]

mycursor.executemany(sql, val)

mydb.commit()

print(mycursor.rowcount, "was inserted.")"""

"""mycursor = mydb.cursor()

mycursor.execute("SELECT * FROM customers")

myresult = mycursor.fetchall()

for x in myresult:
  print(x)"""

"""mycursor = mydb.cursor()

mycursor.execute("CREATE DATABASE mydatabase2")"""

"""mycursor = mydb.cursor()

mycursor.execute("SHOW DATABASES")

for x in mycursor:
  print(x)"""

"""mycursor = mydb.cursor()

mycursor.execute("CREATE TABLE manager (name VARCHAR(255), address VARCHAR(255))")"""

"""mycursor = mydb.cursor()

sql = "INSERT INTO manager (name, address) VALUES (%s, %s)"
val = ("John", "Highway 21")
mycursor.execute(sql, val)

mydb.commit()

print(mycursor.rowcount, "record inserted.")"""

"""mycursor = mydb.cursor()

sql = "INSERT INTO manager (name, address) VALUES (%s, %s)"
val = [
  ('Peter', 'Lowstreet 4'),
  ('Amy', 'Apple st 652'),
  ('Hannah', 'Mountain 21'),
  ('Michael', 'Valley 345'),
  ('Sandy', 'Ocean blvd 2'),
  ('Betty', 'Green Grass 1'),
  ('Richard', 'Sky st 331'),
  ('Susan', 'One way 98'),
  ('Vicky', 'Yellow Garden 2'),
  ('Ben', 'Park Lane 38'),
  ('William', 'Central st 954'),
  ('Chuck', 'Main Road 989'),
  ('Viola', 'Sideway 1633')
]

mycursor.executemany(sql, val)

mydb.commit()

print(mycursor.rowcount, "was inserted.")"""

mycursor = mydb.cursor()

mycursor.execute("SELECT * FROM manager")

myresult = mycursor.fetchall()

for x in myresult:
  print(x)
