from typing import Dict, Optional, Any


async def price_analysis_tool(
    product: str, parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyse the price of a product and return a breakdown.

    Parameters
    ----------
    product : str
        Name or description of the product to analyse.
    parameters : dict
        Additional parameters such as speed, number of IP addresses, etc.
    """
    # Extract parameters with defaults
    speed: int = int(parameters.get("kecepatan_mbps", 50))
    ip_count: int = int(parameters.get("jumlah_ip", 1))
    storage_gb: int = int(parameters.get("hosting_gb", 0))
    domain_count: int = int(parameters.get("jumlah_domain", 0))
    jabatan: Optional[str] = parameters.get("jabatan")

    # Tarif Port berdasarkan bandwidth (IDR per Mbps)
    def tarif_port(speed_mbps: int) -> int:
        if speed_mbps <= 50:
            return 79000
        if speed_mbps <= 500:
            return 70000
        if speed_mbps <= 1000:
            return 63000
        return 57000

    port_price = tarif_port(speed) * speed
    intercity_price = 0  # As per example, 0 for Internet Dedicated
    akses_price = 1_400_000  # FO access default
    instalasi_price = 5_000_000  # Installation cost (OTC)
    ip_price = ip_count * 1_000_000
    # Hosting price table
    hosting_prices = {1: 250_000, 10: 1_545_000}
    hosting_price = hosting_prices.get(storage_gb, 0)
    domain_price = domain_count * 500_000

    subtotal_mrc = (
        akses_price
        + port_price
        + intercity_price
        + ip_price
        + hosting_price
        + domain_price
    )
    subtotal_otc = instalasi_price

    # Diskon berdasarkan jabatan
    diskon_mrc_persen = 0
    diskon_otc_persen = 0
    if jabatan:
        jabatan_lower = jabatan.lower()
        if "manager" in jabatan_lower:
            diskon_mrc_persen = 10
            diskon_otc_persen = 10
        elif "gm" in jabatan_lower:
            diskon_mrc_persen = 15
            diskon_otc_persen = 15

    total_diskon_mrc = subtotal_mrc * diskon_mrc_persen / 100
    total_diskon_otc = subtotal_otc * diskon_otc_persen / 100

    total_mrc_after = subtotal_mrc - total_diskon_mrc
    total_otc_after = subtotal_otc - total_diskon_otc

    return {
        "status": "success",
        "jenis_layanan": product,
        "kecepatan_mbps": speed,
        "akses": {
            "jenis": "FO",
            "biaya_akses": akses_price,
            "biaya_instalasi": instalasi_price,
        },
        "port": {"harga_per_mbps": tarif_port(speed), "total": port_price},
        "intercity": {"harga_per_mbps": 0, "total": intercity_price},
        "ip_public": {"jumlah": ip_count, "harga_per_ip": 1_000_000, "total": ip_price},
        "hosting": {"kapasitas_gb": storage_gb, "harga": hosting_price},
        "domain": {
            "jumlah": domain_count,
            "harga_per_domain": 500_000,
            "total": domain_price,
        },
        "subtotal_mrc": subtotal_mrc,
        "subtotal_otc": subtotal_otc,
        "diskon": {
            "jabatan": jabatan or "",
            "diskon_mrc_persen": diskon_mrc_persen,
            "diskon_otc_persen": diskon_otc_persen,
            "total_diskon_mrc": int(total_diskon_mrc),
            "total_diskon_otc": int(total_diskon_otc),
        },
        "total_setelah_diskon": {
            "mrc": int(total_mrc_after),
            "otc": int(total_otc_after),
        },
        "catatan": "Tarif belum termasuk PPN. Kontrak minimal 12 bulan.",
    }


async def product_calculator_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper around ``price_analysis_tool`` for backward compatibility.

    The ``price_analysis_tool`` expects a product name and parameters.  This
    function extracts those values from ``params`` and forwards them.
    """
    product = params.get("product", "Internet Dedicated")
    return await price_analysis_tool(product, params)
