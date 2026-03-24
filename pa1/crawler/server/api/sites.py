from flask import Blueprint, request, jsonify, abort
from sqlalchemy.exc import IntegrityError
import database.database as database

bp = Blueprint("sites", __name__, url_prefix="/sites")

def to_dict_site(s):
    return {"id": s.id, "domain": s.domain, "robots_content": s.robots_content, "sitemap_content": s.sitemap_content}

@bp.route("/", methods=["GET"])
def list_sites():
    db = database.SessionLocal()
    try:
        sites = db.query(database.Site).all()
        return jsonify([to_dict_site(s) for s in sites])
    finally:
        db.close()

@bp.route("/<int:site_id>", methods=["GET"])
def get_site(site_id):
    db = database.SessionLocal()
    try:
        s = db.query(database.Site).get(site_id)
        if not s:
            abort(404)
        return jsonify(to_dict_site(s))
    finally:
        db.close()

@bp.route("/by-domain", methods=["GET"])
def get_site_by_domain():
    domain = request.args.get("domain")

    if not domain:
        abort(400, description="Missing 'domain' query parameter")

    db = database.SessionLocal()
    try:
        s = db.query(database.Site).filter(database.Site.domain == domain).first()

        if not s:
            abort(404, description="Site not found")

        return jsonify({"id": s.id})
    finally:
        db.close()

@bp.route("/", methods=["POST"])
def create_site():
    payload = request.get_json(force=True)
    db = database.SessionLocal()
    try:
        s = database.Site(domain=payload.get("domain"), robots_content=payload.get("robots_content"), sitemap_content=payload.get("sitemap_content"))
        db.add(s)
        db.commit()
        db.refresh(s)
        return jsonify(to_dict_site(s)), 201
    except IntegrityError:
        db.rollback()
        abort(400)
    finally:
        db.close()

@bp.route("/<int:site_id>", methods=["PUT"])
def update_site(site_id):
    payload = request.get_json(force=True)
    db = database.SessionLocal()
    try:
        s = db.query(database.Site).get(site_id)
        if not s:
            abort(404)
        for k in ("domain", "robots_content", "sitemap_content"):
            if k in payload:
                setattr(s, k, payload[k])
        db.commit()
        return jsonify(to_dict_site(s))
    finally:
        db.close()

@bp.route("/<int:site_id>", methods=["DELETE"])
def delete_site(site_id):
    db = database.SessionLocal()
    try:
        s = db.query(database.Site).get(site_id)
        if not s:
            abort(404)
        db.delete(s)
        db.commit()
        return "", 204
    finally:
        db.close()