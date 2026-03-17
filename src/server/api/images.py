from flask import Blueprint, request, jsonify, abort
from sqlalchemy.exc import IntegrityError
import base64
import database.database as database

bp = Blueprint("images", __name__, url_prefix="/images")

def to_dict_image(i):
    data_b64 = None
    if i.data is not None:
        data_b64 = base64.b64encode(i.data).decode("ascii")
    return {
        "id": i.id,
        "page_id": i.page_id,
        "filename": i.filename,
        "content_type": i.content_type,
        "data": data_b64,
        "accessed_time": i.accessed_time,
    }

@bp.route("/", methods=["GET"])
def list_images():
    db = database.SessionLocal()
    try:
        items = db.query(database.Image).all()
        return jsonify([to_dict_image(i) for i in items])
    finally:
        db.close()

@bp.route("/<int:image_id>", methods=["GET"])
def get_image(image_id):
    db = database.SessionLocal()
    try:
        i = db.query(database.Image).get(image_id)
        if not i:
            abort(404)
        return jsonify(to_dict_image(i))
    finally:
        db.close()

@bp.route("/", methods=["POST"])
def create_image():
    db = database.SessionLocal()
    try:
        if request.files and "file" in request.files:
            f = request.files["file"]
            data_bytes = f.read()
            filename = f.filename or request.form.get("filename")
            content_type = f.content_type or request.form.get("content_type")
            page_id = request.form.get("page_id")
            accessed_time = request.form.get("accessed_time")
            page_id_val = int(page_id) if page_id is not None else None

            i = database.Image(
                page_id=page_id_val,
                filename=filename,
                content_type=content_type,
                data=data_bytes,
                accessed_time=accessed_time,
            )

        else:
            payload = request.get_json(force=True)
            data_b64 = payload.get("data")
            data_bytes = base64.b64decode(data_b64) if data_b64 else None

            i = database.Image(
                page_id=payload.get("page_id"),
                filename=payload.get("filename"),
                content_type=payload.get("content_type"),
                data=data_bytes,
                accessed_time=payload.get("accessed_time"),
            )

        db.add(i)
        db.commit()
        db.refresh(i)
        return jsonify(to_dict_image(i)), 201
    except IntegrityError:
        db.rollback()
        abort(400)
    finally:
        db.close()

@bp.route("/<int:image_id>", methods=["PUT"])
def update_image(image_id):
    db = database.SessionLocal()
    try:
        i = db.query(database.Image).get(image_id)
        if not i:
            abort(404)

        # multipart update (file + form fields)
        if request.files and "file" in request.files:
            f = request.files["file"]
            i.data = f.read()
            if f.filename:
                i.filename = f.filename
            if f.content_type:
                i.content_type = f.content_type

        # json or form fields for metadata
        if request.content_type and request.content_type.startswith("multipart/"):
            form = request.form
            if "page_id" in form:
                i.page_id = int(form.get("page_id")) if form.get("page_id") else None
            if "filename" in form:
                i.filename = form.get("filename")
            if "content_type" in form:
                i.content_type = form.get("content_type")
            if "accessed_time" in form:
                i.accessed_time = form.get("accessed_time")
        else:
            payload = request.get_json(silent=True)
            if payload:
                for k in ("page_id", "filename", "content_type", "accessed_time"):
                    if k in payload:
                        setattr(i, k, payload[k])
                if "data" in payload:
                    i.data = base64.b64decode(payload["data"]) if payload["data"] else None

        db.commit()
        return jsonify(to_dict_image(i))
    finally:
        db.close()

@bp.route("/<int:image_id>", methods=["DELETE"])
def delete_image(image_id):
    db = database.SessionLocal()
    try:
        i = db.query(database.Image).get(image_id)
        if not i:
            abort(404)
        db.delete(i)
        db.commit()
        return "", 204
    finally:
        db.close()