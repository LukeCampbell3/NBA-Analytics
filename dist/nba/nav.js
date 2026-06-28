(() => {
    const body = document.body;
    const root = document.documentElement;
    const menuBtn = document.getElementById('siteMenuButton');
    const drawer = document.getElementById('siteNavDrawer');
    const overlay = document.getElementById('siteNavOverlay');
    const closeBtn = document.getElementById('siteNavClose');
    const themeToggleBtn = document.getElementById('themeToggleButton');
    const THEME_KEY = 'sports_site_theme_preference';

    const getPreferredTheme = () => {
        const saved = localStorage.getItem(THEME_KEY);
        if (saved === 'dark' || saved === 'light') return saved;
        return (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light';
    };

    const applyTheme = (theme) => {
        const isDark = theme === 'dark';
        body.classList.toggle('theme-dark', isDark);
        root.classList.toggle('theme-dark', isDark);
        window.dispatchEvent(new CustomEvent('themechange', { detail: { theme } }));
        if (themeToggleBtn) {
            themeToggleBtn.textContent = isDark ? 'L' : 'D';
            themeToggleBtn.title = isDark ? 'Switch to light mode' : 'Switch to dark mode';
            themeToggleBtn.setAttribute('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
        }
    };

    applyTheme(getPreferredTheme());

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const next = body.classList.contains('theme-dark') ? 'light' : 'dark';
            localStorage.setItem(THEME_KEY, next);
            applyTheme(next);
        });
    }

    // Optional drawer wiring (some pages do not include drawer markup).
    if (!menuBtn || !drawer || !overlay) return;

    const openNav = () => {
        drawer.classList.add('open');
        overlay.classList.add('open');
        body.classList.add('nav-open');
        menuBtn.setAttribute('aria-expanded', 'true');
    };

    const closeNav = () => {
        drawer.classList.remove('open');
        overlay.classList.remove('open');
        body.classList.remove('nav-open');
        menuBtn.setAttribute('aria-expanded', 'false');
    };

    const isOpen = () => drawer.classList.contains('open');

    menuBtn.addEventListener('click', () => {
        if (isOpen()) closeNav();
        else openNav();
    });

    if (closeBtn) closeBtn.addEventListener('click', closeNav);
    overlay.addEventListener('click', closeNav);

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && isOpen()) closeNav();
    });

    const normalizePath = (value) => {
        const url = new URL(value || '/', window.location.origin);
        let path = url.pathname.toLowerCase().replace(/\/index\.html$/, '/');
        path = path.replace(/\/+$/, '/');
        return path || '/';
    };

    const currentPath = normalizePath(window.location.pathname);
    const links = Array.from(document.querySelectorAll('.site-nav-link'));
    let bestActiveLink = null;
    let bestActiveLength = -1;

    links.forEach((link) => {
        const href = normalizePath(link.getAttribute('href') || '/');
        const isActive = href === '/'
            ? currentPath === '/'
            : currentPath === href || currentPath.startsWith(href);

        if (isActive && href.length > bestActiveLength) {
            bestActiveLink = link;
            bestActiveLength = href.length;
        }
        link.addEventListener('click', closeNav);
    });

    if (bestActiveLink) {
        bestActiveLink.classList.add('active');
        bestActiveLink.setAttribute('aria-current', 'page');
    }
})();
