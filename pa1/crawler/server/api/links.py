from flask import Blueprint, request, jsonify, abort
from sqlalchemy.exc import IntegrityError
import database.database as database
from sqlalchemy.orm import aliased

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

@bp.route("/with-urls/", methods=["GET"])
def list_links_with_urls():
    db = database.SessionLocal()
    try:
        p1 = aliased(database.Page)
        p2 = aliased(database.Page)

        items = (
            db.query(
                database.Link.from_page,
                database.Link.to_page,
                p1.url.label("from_url"),
                p2.url.label("to_url"),
                p1.accessed_time.label("time")
            )
            .join(p1, database.Link.from_page == p1.id)
            .join(p2, database.Link.to_page == p2.id)
            .all()
        )

        return jsonify([
            {
                "from_page": row.from_page,
                "to_page": row.to_page,
                "from_url": row.from_url,
                "to_url": row.to_url,
                "accessed_time": row.time
            }
            for row in items
        ])

    except IntegrityError as e:
        print(e, flush=True)
    finally:
        db.close()
