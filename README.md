# Document Quality Validation API

A minimal PHP 8.3 API whose only job is to forward an uploaded document photo
to an external Image Quality Validation API and relay its ACCEPTED/REJECTED
JSON response back to the caller. No OCR, no database, no image storage, no
authentication — the uploaded file is deleted immediately after the external
API responds (or if any error occurs).

## Files

| File                | Responsibility                                                              |
|---------------------|-------------------------------------------------------------------------------|
| `index.php`         | Front controller / REST endpoint. Reads `$_FILES`, returns JSON.             |
| `ImageValidator.php`| Validates the upload locally (present, real image, size, mime), calls `ApiClient`, guarantees cleanup of the temp file. |
| `ApiClient.php`     | cURL client that forwards the file to the external API as `multipart/form-data` and returns its decoded JSON. |
| `config.php`        | Loads settings from environment variables / `.env`.                          |

## Setup

```bash
cp .env.example .env
```

Edit `.env`:

```
EXTERNAL_API_URL=https://your-image-quality-api.example.com/validate
EXTERNAL_API_KEY=your-api-key-if-required
```

No Composer dependencies are required — everything uses core PHP extensions
(`curl`, `fileinfo`, `json`).

## Running locally

```bash
php -S localhost:8080
```

## Usage

`POST /index.php` (or `/` if using the included `.htaccess` on Apache) with
`multipart/form-data`, field name `image`:

```bash
curl -X POST http://localhost:8080/index.php \
  -F "image=@/path/to/photo.jpg"
```

The response is exactly what the external API returned, passed through
unchanged, e.g.:

```json
{
    "status": "accepted",
    "score": 95,
    "message": "Image quality is sufficient."
}
```

or

```json
{
    "status": "rejected",
    "score": 58,
    "message": "Please retake the photo.",
    "reasons": [
        "Image is blurry",
        "Low brightness",
        "Document is partially cropped"
    ]
}
```

## Local error responses

If the request never reaches the external API (bad method, missing file,
oversized file, wrong file type, or the external API is unreachable/invalid),
the API returns its own JSON error instead:

| HTTP Status | Cause                                                     |
|-------------|-------------------------------------------------------------|
| 400         | No file, invalid file, too large, or unsupported mime type |
| 405         | Method other than `POST`                                    |
| 500         | `EXTERNAL_API_URL` not configured                            |
| 502         | External API unreachable, non-2xx, or returned invalid JSON |

```json
{ "status": "error", "message": "Uploaded file is not a valid image." }
```

## Guarantees

- Nothing is written to a database.
- The uploaded file is never moved to permanent storage.
- The PHP-managed temp file is `unlink()`-ed in a `finally` block right after
  the external API call completes (success or failure) — not left for
  end-of-request garbage collection.
- No authentication layer — add one at the infrastructure level (API gateway,
  mTLS, IP allowlist) if the API needs to be exposed publicly.
