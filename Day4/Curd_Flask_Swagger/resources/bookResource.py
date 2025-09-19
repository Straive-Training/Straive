from flask_restful import Resource
from flask import request
import json
from util.common import get_db_connection


class BooksGETResource(Resource):
    def get(self):
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM books")
        books = cursor.fetchall()
        cursor.close()
        conn.close()
        return books


class BookGETResource(Resource):
    def get(self, id):
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM books WHERE id = %s", (id,))
        book = cursor.fetchone()
        cursor.close()
        conn.close()
        if book:
            return book
        return {"message": "Book not found"}, 404


class BookPOSTResource(Resource):
    def post(self):
        data = json.loads(request.data)
        title = data.get("title")
        if not title:
            return {"message": "Title is required"}, 400

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO books (title) VALUES (%s)", (title,))
        conn.commit()
        new_id = cursor.lastrowid
        cursor.close()
        conn.close()

        return {"id": new_id, "title": title}, 201


class BookPUTResource(Resource):
    def put(self, id):
        data = json.loads(request.data)
        title = data.get("title")
        if not title:
            return {"message": "Title is required"}, 400

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM books WHERE id = %s", (id,))
        if cursor.fetchone() is None:
            cursor.close()
            conn.close()
            return {"message": "Book not found"}, 404

        cursor.execute("UPDATE books SET title = %s WHERE id = %s", (title, id))
        conn.commit()
        cursor.close()
        conn.close()

        return {"id": id, "title": title}, 200


class BookDELETEResource(Resource):
    def delete(self, id):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM books WHERE id = %s", (id,))
        if cursor.fetchone() is None:
            cursor.close()
            conn.close()
            return {"message": "Book not found"}, 404

        cursor.execute("DELETE FROM books WHERE id = %s", (id,))
        conn.commit()
        cursor.close()
        conn.close()

        return "", 204
