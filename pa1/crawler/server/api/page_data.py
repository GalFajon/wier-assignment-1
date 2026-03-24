from flask import Blueprint, request, jsonify, abort
from sqlalchemy.exc import IntegrityError
import base64
import database.database as database

bp = Blueprint("page_data", __name__, url_prefix="/page_data")

def to_dict_page_data(d):
    data_b64 = None
    if d.data is not None:
        data_b64 = base64.b64encode(d.data).decode("ascii")
    return {"id": d.id, "page_id": d.page_id, "data_type_code": d.data_type_code, "data": data_b64}

@bp.route("/", methods=["GET"])
def list_page_data():
    db = database.SessionLocal()
    try:
        items = db.query(database.PageData).all()
        return jsonify([to_dict_page_data(i) for i in items])
    finally:
        db.close()

@bp.route("/<int:data_id>", methods=["GET"])
def get_page_data(data_id):
    db = database.SessionLocal()
    try:
        d = db.query(database.PageData).get(data_id)
        if not d:
            abort(404)
        return jsonify(to_dict_page_data(d))
    finally:
        db.close()

@bp.route("/", methods=["POST"])
def create_page_data():
    # Support multipart/form-data file upload or JSON(base64)
    db = database.SessionLocal()
    try:
        if request.files and "file" in request.files:
            f = request.files["file"]
            data_bytes = f.read()
            page_id = request.form.get("page_id")
            data_type_code = request.form.get("data_type_code")
            page_id_val = int(page_id) if page_id is not None and page_id != "" else None
            d = database.PageData(page_id=page_id_val, data_type_code=data_type_code, data=data_bytes)
        else:
            payload = request.get_json(force=True)
            data_b64 = payload.get("data")
            data_bytes = base64.b64decode(data_b64) if data_b64 else None
            d = database.PageData(page_id=payload.get("page_id"), data_type_code=payload.get("data_type_code"), data=data_bytes)

        db.add(d)
        db.commit()
        db.refresh(d)
        return jsonify(to_dict_page_data(d)), 201
    except IntegrityError:
        db.rollback()
        abort(400)
    finally:
        db.close()

@bp.route("/<int:data_id>", methods=["PUT"])
def update_page_data(data_id):
    db = database.SessionLocal()
    try:
        d = db.query(database.PageData).get(data_id)
        if not d:
            abort(404)

        # multipart update (file + form fields)
        if request.files and "file" in request.files:
            f = request.files["file"]
            d.data = f.read()

        # handle form fields (multipart) or JSON
        if request.content_type and request.content_type.startswith("multipart/"):
            form = request.form
            if "page_id" in form:
                d.page_id = int(form.get("page_id")) if form.get("page_id") else None
            if "data_type_code" in form:
                d.data_type_code = form.get("data_type_code")
        else:
            payload = request.get_json(silent=True)
            if payload:
                if "page_id" in payload:
                    d.page_id = payload["page_id"]
                if "data_type_code" in payload:
                    d.data_type_code = payload["data_type_code"]
                if "data" in payload:
                    d.data = base64.b64decode(payload["data"]) if payload["data"] else None

        db.commit()
        return jsonify(to_dict_page_data(d))
    finally:
        db.close()

@bp.route("/<int:data_id>", methods=["DELETE"])
def delete_page_data(data_id):
    db = database.SessionLocal()
    try:
        d = db.query(database.PageData).get(data_id)
        if not d:
            abort(404)
        db.delete(d)
        db.commit()
        return "", 204
    finally:
        db.close()