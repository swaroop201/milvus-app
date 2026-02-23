# Static Assets

This directory contains static files for the Chat RAG Assistant application.

## Structure

```
static/
└── css/
    └── style.css    # Main stylesheet with responsive design
```

## CSS Features

The `style.css` file includes:

- **Design System**: CSS custom properties (variables) for colors, fonts, spacing, and animations
- **Responsive Layout**: Mobile-first design with breakpoints for different screen sizes
- **Accessibility**: Focus states, reduced motion preferences, and semantic color contrast
- **Dark Theme**: Modern dark mode with amber accent colors
- **Animations**: Smooth transitions and keyframe animations for messages
- **Components**:
  - Header with gradient text
  - Ingest card with icon and description
  - Mode tabs (Chat/Ask RAG)
  - Chat container with scrollable messages
  - Input area with send button
  - Empty state
  - Message styling for user/assistant/error states
  - Typing indicator with animated dots

## Customization

All colors, fonts, and spacing are defined in CSS custom properties at the `:root` level. You can easily customize:

```css
:root {
  --color-bg-primary: #0c0c0f;
  --color-amber-primary: #f59e0b;
  --font-primary: "Outfit", system-ui, sans-serif;
  /* ... more variables ... */
}
```

## Browser Support

- Modern browsers with CSS Grid, Flexbox, and CSS custom properties support
- Mobile responsive (320px and up)
- Supports dark mode preferences
- Webkit-based browsers for smooth animations
