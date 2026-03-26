from flask import Blueprint, request, jsonify, abort
from sqlalchemy.exc import IntegrityError
from sqlalchemy import null
import database.database as database
from sqlalchemy.dialects.postgresql import insert

bp = Blueprint("pages", __name__, url_prefix="/pages")

def to_dict_page(p):
    return {
        "id": p.id,
        "site_id": p.site_id,
        "page_type_code": p.page_type_code,
        "url": p.url,
        "html_content": p.html_content,
        "http_status_code": p.http_status_code,
        "content_hash": p.content_hash,
        "priority": p.priority,
        "accessed_time": p.accessed_time,
    }

@bp.route("/", methods=["GET"])
def list_pages():
    db = database.SessionLocal()
    try:
        pages = db.query(database.Page).all()
        return jsonify([to_dict_page(p) for p in pages])
    finally:
        db.close()

@bp.route("/<int:page_id>", methods=["GET"])
def get_page(page_id):
    db = database.SessionLocal()
    try:
        p = db.query(database.Page).get(page_id)
        if not p:
            abort(404)
        return jsonify(to_dict_page(p))
    finally:
        db.close()


@bp.route("/", methods=["POST"])
def create_page():
    payload = request.get_json(force=True)
    db = database.SessionLocal()

    # if the payload is an array, handle as frontier pages
    if type(payload) is list:
        try:
            pages = []
            for f_page in payload:
                p = database.Page(
                    site_id=f_page.get("site_id"),
                    page_type_code="FRONTIER",
                    url=f_page.get("url"),
                    html_content=null(),
                    http_status_code=null(),
                    content_hash=null(),
                    priority=f_page.get("priority"),
                    accessed_time=null(),
                )
                pages.append(p)
                db.add(p)
            db.commit()
            p_dicts = []
            for p in pages:
                db.refresh(p)
                p_dicts.append(to_dict_page(p))
            return jsonify({}), 201
        except IntegrityError:
            db.rollback()
            abort(400)
        finally:
            db.close()
    else:
        try:
            p = database.Page(
                site_id=payload.get("site_id"),
                page_type_code=payload.get("page_type_code"),
                url=payload.get("url"),
                html_content=payload.get("html_content"),
                http_status_code=payload.get("http_status_code"),
                content_hash=payload.get("content_hash"),
                priority=payload.get("priority"),
                accessed_time=payload.get("accessed_time"),
            )
            db.add(p)
            db.commit()
            db.refresh(p)
            return jsonify(to_dict_page(p)), 201
        except IntegrityError:
            db.rollback()
            abort(400)
        finally:
            db.close()
        
@bp.route("/<int:page_id>", methods=["PUT"])
def update_page(page_id):
    payload = request.get_json(force=True)
    db = database.SessionLocal()
    try:
        p = db.query(database.Page).get(page_id)
        if not p:
            abort(404)
        for k in ("site_id", "page_type_code", "url", "html_content", "http_status_code", "content_hash", "priority", "accessed_time"):
            if k in payload:
                setattr(p, k, payload[k])
        db.commit()
        return jsonify(to_dict_page(p))
    finally:
        db.close()

@bp.route("/<int:page_id>", methods=["DELETE"])
def delete_page(page_id):
    db = database.SessionLocal()
    try:
        p = db.query(database.Page).get(page_id)
        if not p:
            abort(404)
        db.delete(p)
        db.commit()
        return "", 204
    finally:
        db.close()

@bp.route("/by-url", methods=["GET"])
def get_page_by_url():
    url = request.args.get("url")

    if not url:
        abort(400, description="Missing 'url' parameter")

    db = database.SessionLocal()
    try:
        p = db.query(database.Page).filter(database.Page.url == url).first()

        if not p:
            abort(404)

        return jsonify(to_dict_page(p))
    finally:
        db.close()