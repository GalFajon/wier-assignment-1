from flask import Blueprint, request, jsonify, abort
from sqlalchemy.exc import IntegrityError
import database.database as database

bp = Blueprint("links", __name__, url_prefix="/links")

def to_dict_link(l):
    return {"from_page": l.from_page, "to_page": l.to_page}

@bp.route("/", methods=["GET"])
def list_links():
    db = database.SessionLocal()
    try:
        items = db.query(database.Link).all()
        return jsonify([to_dict_link(l) for l in items])
    finally:
        db.close()

@bp.route("/", methods=["POST"])
def create_link():
    payload = request.get_json(force=True)
    db = database.SessionLocal()
    try:
        l = database.Link(from_page=payload.get("from_page"), to_page=payload.get("to_page"))
        db.add(l)
        db.commit()
        return jsonify(to_dict_link(l)), 201
    except IntegrityError:
        db.rollback()
        abort(400)
    finally:
        db.close()

@bp.route("/<int:from_page>/<int:to_page>", methods=["DELETE"])
def delete_link(from_page, to_page):
    db = database.SessionLocal()
    try:
        l = db.query(database.Link).get((from_page, to_page))
        if not l:
            abort(404)
        db.delete(l)
        db.commit()
        return "", 204
    finally:
        db.close()